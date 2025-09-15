import geobench
import kornia as K
from kornia.constants import Resample 
import torch
from torchvision.transforms import v2
import numpy as np
# export

class SegDataAugmentation(torch.nn.Module):
    def __init__(self, mean, std, size, split="valid"):
        super().__init__()

        self.norm = K.augmentation.Normalize(mean=mean, std=std)

        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=False),
                K.augmentation.RandomRotation(degrees=90, p=0.5, align_corners=False),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=False),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, x, y):
        x_out, y_out = self.transform(x, y)
        x_out = self.norm(x_out)
        return x_out, y_out


class SegGeoBenchTransform(object):
    def __init__(self, task, split, size, mean, std, band_names=None):
        self.band_names = band_names
        band_stats = task.get_dataset(band_names=band_names).band_stats
        self.max_bands = np.array([band_stats[b].max for b in band_names])
        self.min_bands = np.array([band_stats[b].min for b in band_names])
        print(self.min_bands, self.max_bands)
        self.transform = SegDataAugmentation(mean=mean, std=std, size=size, split=split)

    def __call__(self, sample):
        array, band_names = sample.pack_to_3d(
            band_names=self.band_names, resample=True, fill_value=None, resample_order=3
        )  # h,w,c
        array = (array - self.min_bands) / (self.max_bands - self.min_bands)
        array = torch.from_numpy(array.astype("float32")).permute(2, 0, 1)
        mask = torch.from_numpy(sample.label.data.astype("uint8")).squeeze(-1)
        array, mask = self.transform(array.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))

        return array.squeeze(0), mask.squeeze(0).squeeze(0)


class GeoBenchDataset:
    def __init__(self, config):
        self.dataset_config = config
        task_iter = geobench.task_iterator(benchmark_name=config.benchmark_name)
        self.tasks = {task.dataset_name: task for task in task_iter}
        self.img_size = (config.image_resolution, config.image_resolution)
        self.mean = torch.Tensor(config.mean)
        self.std = torch.Tensor(config.std)

    def create_dataset(self):
        task = self.tasks.get(self.dataset_config.dataset_name)
        assert task is not None, (
            f"{self.dataset_config.dataset_name} doesn't exist in geobench"
        )
        GeoBenchTransform = SegGeoBenchTransform

        train_transform = GeoBenchTransform(
            task,
            split="train",
            size=self.img_size,
            mean=self.mean,
            std=self.std,
            band_names=self.dataset_config.band_names,
        )
        val_transform = GeoBenchTransform(
            task,
            split="valid",
            size=self.img_size,
            mean=self.mean,
            std=self.std,
            band_names=self.dataset_config.band_names,
        )

        dataset_train = task.get_dataset(
            split="train",
            transform=train_transform,
            band_names=self.dataset_config.band_names,
        )
        dataset_val = task.get_dataset(
            split="valid",
            transform=val_transform,
            band_names=self.dataset_config.band_names,
        )
        dataset_test = task.get_dataset(
            split="test",
            transform=val_transform,
            band_names=self.dataset_config.band_names,
        )

        return dataset_train, dataset_val, dataset_test

class BenchmarkDataModule():
    def __init__(self, dataset_config, batch_size, num_workers, pin_memory):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.dataset_train, self.dataset_val, self.dataset_test = \
            GeoBenchDataset(self.dataset_config).create_dataset()
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )