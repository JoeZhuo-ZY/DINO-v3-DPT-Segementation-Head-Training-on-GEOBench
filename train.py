import os
import time
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dpt_head import DPTHead
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from torchvision import transforms
from omegaconf import OmegaConf
from PIL import Image

from geobench_wrapper import GeoBenchDataset, BenchmarkDataModule

import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior.*")

def inspect_dataloader(dl, args):
    img, mask = next(iter(dl))
    print(f"Image shape: {img.shape}, mask shape: {mask.shape}")
    # Print min/max per channel for the first image
    for c in range(img.shape[1]):
        print(f"Channel {c}: min={img[:,c,:,:].min().item():.4f}, max={img[:,c,:,:].max().item():.4f}")
    # Save the first image and mask with renormalization
    out_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Renormalize image to [0, 1] for saving
    mean = torch.tensor(args.mean).view(1, -1, 1, 1)
    std = torch.tensor(args.std).view(1, -1, 1, 1)
    img_renorm = img[-1].cpu() * std[0] + mean[0]
    img_renorm = img_renorm.clamp(0, 1)
    img_pil = transforms.ToPILImage()(img_renorm)
    img_pil.save(out_dir / f"{args.task}_sample_image.png")

    # Save mask as PNG (convert to uint8 for visualization)
    mask_pil = Image.fromarray(mask[0].cpu().numpy().astype("uint8"))
    mask_pil.save(out_dir / f"{args.task}_sample_mask.png")

# =========================
#   Models
# =========================
class ViT_DPT_Seg(nn.Module):
    def __init__(self, vit_encoder, dpt_head: DPTHead):
        super().__init__()
        self.encoder = vit_encoder
        self.head = dpt_head

    @torch.no_grad()
    def _encode(self, x):
        return self.encoder(x)

    def forward(self, pixel_values: torch.Tensor):
        features = self._encode(pixel_values)
        logits = self.head(features)
        return logits


class HFBackboneWrapper(nn.Module):
    """
    Wraps a pretrained HF model and exposes a custom forward.
    """
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        return_hidden_states: bool = True,
        layers=(-4, -3, -2, -1),
        device: torch.device | None = None,
        device_map: str | None = "auto"  # "auto" or None
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        if return_hidden_states:
            cfg.output_hidden_states = True

        # Load model
        if device_map == "auto":
            self.model = AutoModel.from_pretrained(model_name, config=cfg, device_map="auto")
        else:
            self.model = AutoModel.from_pretrained(model_name, config=cfg)
            if device is not None:
                self.model.to(device)

        self.layers = layers  # picks from hidden_states; negative indices are supported
        self.patch_size = getattr(cfg, "patch_size", 16)  # ViT configs have this
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 4)  # default 4 if absent

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, pixel_values: torch.Tensor):
        """
        Returns: list of tuples per chosen layer:
            [(feat, register_tok or None, cls_tok), ...]
        - feat: [B, C, H', W']  (H' = H/ph, W' = W/pw)
        - register_tok: [B, R, C] or None
        - cls_tok: [B, C]
        """
        B, C, H, W = pixel_values.shape

        # support int or tuple patch sizes
        if isinstance(self.patch_size, int):
            ph = pw = self.patch_size
        else:
            ph, pw = self.patch_size

        assert H % ph == 0 and W % pw == 0, f"Input {H}x{W} not divisible by patch {ph}x{pw}"
        H_, W_ = H // ph, W // pw

        out = self.model(pixel_values)  # returns object with .hidden_states
        hs = out.hidden_states          # tuple(len = num_layers+1), hs[0] = embeddings
        R = self.num_register_tokens    # number of register tokens, e.g. 4

        outs = []
        for li in self.layers:
            tokens = hs[li]             # [B, 1+R+N, C]
            cls_tok = tokens[:, 0]      # [B, C]
            reg_tok = tokens[:, 1:1+R] if R > 0 else None
            patch_tok = tokens[:, 1+R:] # [B, N, C], N = H_*W_

            # sanity check
            N = patch_tok.shape[1]
            assert N == H_ * W_, f"Token count {N} != H_*W_ = {H_}*{W_}"

            # reshape to feature map [B, C, H', W']
            patch_tok = patch_tok.reshape(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()
            outs.append((patch_tok, reg_tok, cls_tok))

        return outs


# =========================
#   Utils
# =========================
def up_to_mask_size(logits, mask):
    return logits if logits.shape[-2:] == mask.shape[-2:] \
           else F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)

def prepare_targets(mask):
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    return mask.long() if mask.dtype != torch.long else mask

@torch.no_grad()
def validate(model, loader, num_classes, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    agg = defaultdict(float)
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        logits = up_to_mask_size(model(img), mask)
        tgt = prepare_targets(mask)
        loss_sum += ce(logits, tgt).item() * img.size(0); n += img.size(0)

        pred = logits.argmax(1)
        valid = (tgt >= 0) & (tgt < num_classes)
        total = valid.sum().item() + 1e-6
        correct = (pred[valid] == tgt[valid]).sum().item()
        pix_acc = correct / total

        ious = []
        for c in range(num_classes):
            p, t = (pred == c) & valid, (tgt == c) & valid
            inter = (p & t).sum().item()
            union = (p | t).sum().item() + 1e-6
            ious.append(inter / union)
        agg["pixel_acc"] += pix_acc * img.size(0)
        agg["miou"] += (sum(ious) / len(ious)) * img.size(0)
    return {"val_loss": loss_sum / n, "pixel_acc": agg["pixel_acc"] / n, "miou": agg["miou"] / n}


def train(
    model,
    train_loader,
    val_loader,
    num_classes=2,
    lr=3e-4, #[3e−5,1e−4,3e−4,1e−3]
    weight_decay=1e-4,
    max_steps=10_000,
    val_interval=100,
    ckpt_dir="./ckpts",
    grad_clip=1.0,
    print_every=100,
    task="pv4ger_seg",
    project_name="GeoBench-Dinov3",
    use_wandb=True,
    device=None,
):
    if use_wandb:
        wandb.init(project=project_name, name=f"geobench_10layers_512_{task}_lr{lr}",
                   config={"lr": lr, "weight_decay": weight_decay, "max_steps": max_steps,
                           "val_interval": val_interval})
    model.train()
    ckpt_dir = Path(wandb.run.dir) if use_wandb else Path(ckpt_dir)
    ce = nn.CrossEntropyLoss()
    params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best_miou, step = -1.0, 0
    pbar = tqdm(total=max_steps, desc=f"Train (LR={lr:.1e})", ncols=96)
    running = 0.0
    it = iter(train_loader)

    while step < max_steps:
        try:
            img, mask = next(it)
        except StopIteration:
            it = iter(train_loader)
            img, mask = next(it)

        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Always size-match to target before loss
        logits = up_to_mask_size(model(img), mask)
        tgt = prepare_targets(mask)
        loss = ce(logits, tgt)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        step += 1
        running += loss.item()
        pbar.update(1)

        if step % print_every == 0:
            avg_loss = running / print_every
            if use_wandb:
                wandb.log({"train_loss": avg_loss, "step": step})
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            running = 0.0

        if step % val_interval == 0 or step == max_steps:
            valm = validate(model, val_loader, num_classes, device)
            if use_wandb:
                wandb.log({"val_loss": valm["val_loss"], "val_mIoU": valm["miou"],
                           "val_pixelAcc": valm["pixel_acc"], "step": step})

            pbar.write(f"[{step}] val_loss={valm['val_loss']:.4f} "
                       f"mIoU={valm['miou']:.4f} pixAcc={valm['pixel_acc']:.4f}")
            torch.save(model.state_dict(), f"{ckpt_dir}/last.pt")
            if valm["miou"] > best_miou:
                best_miou = valm["miou"]
                torch.save(model.state_dict(), f"{ckpt_dir}/best_weights.pt")
                pbar.write(f"✅ New best mIoU={best_miou:.4f} saved.")

    pbar.close()
    if use_wandb:
        wandb.finish()
    return best_miou, f"{ckpt_dir}/best_weights.pt"


@torch.no_grad()
def evaluate_on_test(model, test_loader, num_classes=2, device=None, ckpt="./ckpts/best_weights.pt"):
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd if not isinstance(sd, dict) or "state_dict" not in sd else sd["state_dict"])
    model.eval()
    return validate(model, test_loader, num_classes, device)


# =========================
#   CLI / Main
# =========================
def parse_layers(s: str):
    # e.g. "-4,-3,-2,-1"
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

def main():
    parser = argparse.ArgumentParser(description="GeoBench segmentation with ViT+DPT")
    # Data
    parser.add_argument("--dataconfig-path", type=str,
                        help="Path to dataset OmegaConf YAML")
    parser.add_argument("--task", type=str, default="pv4ger_seg",)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    parser.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225))

    # Model / Backbone
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--freeze", action="store_true", default=True)
    parser.add_argument("--no-freeze", dest="freeze", action="store_false")
    parser.add_argument("--layers", type=parse_layers, default="-4,-3,-2,-1",
                        help="Comma-separated hidden-state indices, supports negatives")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Train
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--ckpt-dir", type=str, default="./ckpts")

    # Logging
    parser.add_argument("--project-name", type=str, default="GeoBench-Dinov3")
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")

    # Repro
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # =========================
    # Data
    # =========================
    config_path = args.dataconfig_path + f"/geobench_{args.task}.yaml"
    config_data = OmegaConf.load(config_path)
    config_data.image_resolution = int(args.image_resolution)
    config_data.mean = tuple(args.mean)
    config_data.std = tuple(args.std)

    train_loader, val_loader, test_loader = BenchmarkDataModule(
        config_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    ).setup()
    class_number = config_data.num_classes

    # =========================
    # Model
    # =========================
    backbone = HFBackboneWrapper(
        model_name=args.model_name,
        freeze=args.freeze,
        return_hidden_states=True,
        layers=args.layers if isinstance(args.layers, tuple) else parse_layers(args.layers),
        device=device,
        device_map="auto"
    )

    seg_model = ViT_DPT_Seg(
        vit_encoder=backbone,
        dpt_head=DPTHead(
            in_channels=[1024, 1024, 1024, 1024],  # DINOv3-ViT-L/16
            embed_dims=1024,
            seg_cls_number=class_number,
            use_sync_bn=False,
        ),
    ).to(device)

    # Sanity prints
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # inspect_dataloader(train_loader, args)

    # =========================
    # Run training
    # =========================
    best_miou, best_ckpt = train(
        seg_model, train_loader, val_loader,
        num_classes=class_number,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        val_interval=args.val_interval,
        ckpt_dir= args.ckpt_dir,
        grad_clip=args.grad_clip,
        print_every=args.print_every,
        project_name=args.project_name,
        task=args.task,
        use_wandb=args.wandb,
        device=device
    )
    print("Best mIoU:", best_miou)

    # =========================
    # Final test eval
    # =========================
    test_metrics = evaluate_on_test(
        seg_model, test_loader,
        num_classes=class_number,
        device=device,
        ckpt=best_ckpt,
    )
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
