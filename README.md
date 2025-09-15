### Minimal reproduction of DINOv3 (ViT) encoder + DPT head for 6 segmentation tasks on GeoBench.
The official training code for SegHead was not released; this repo provides a training loop, a DPT decoder head, and a GeoBench wrapper that supports single-task and multi-task runs.

The DPT head code is from a Meta [repo](https://github.com/facebookresearch/HighResCanopyHeight/blob/main/models/dpt_head.py#L245).

The GeoBench dataset wrapper is from DOFA [repo](https://github.com/xiong-zhitong/DOFA-pytorch/blob/geofm/src/datasets/geobench_wrapper.py).

#### GeoBench Prepocess
We first normalize samples to [0,1] per channel based on the min and max stats provided by GeoBench：

```
band_stats = task.get_dataset(band_names=band_names).band_stats
self.max_bands = np.array([band_stats[b].max for b in band_names])
self.min_bands = np.array([band_stats[b].min for b in band_names])

array = (array - self.min_bands) / (self.max_bands - self.min_bands)
```

and then normalize by the mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
```
self.norm = K.augmentation.Normalize(mean=mean, std=std)
```

#### Main Hyperparameters We Used
* learning-rate 3e-5 (grid search [3e−5,1e−4,3e−4,1e−3] in paper)
* batch-size 8 (32 in paper)
* iteration number 10k (40K in paper)
* Blocks extracted for DPT -4,-3,-2,-1 (not mentioned in paper)
* DINO backbone vitl16-pretrain-lvd1689m (web)
* Input size 512x512 (same as paper)

#### Reproduction Results
| Task / Metric | **Ours** | **Paper – DINOv3 Sat (ViT-L)** | **Δ (pp)** |
| ------------- | -----------: | ---------------------------------: | ---------: |
| m-cashew      |    **41.2** |                               94.2   | (different classes)  |
| m-chesapeake  |     **60.8** |                               75.6 |  **-14.8** |
| m-NeonTree    |     64.5 |                               61.8 |   +2.7 |
| m-nz-cattle   |     81.2 |                               83.7 |  -2.5 |
| m-pv4ger-seg  |     94.8 |                               95.2 |   -0.4 |
| m-SA-crop     |     **28.3** |                               36.8 |   **-8.5** |

The experiments are run on a single NVIDIA 4090 24G. Due to the computation limition, I cannot fully reproduce the setting in the paper.

It still remains significant gap on ``m-cashew`` ``m-chesapeake`` ``m-SA-crop``. I am not sure what is cause (might be prepocess pipeline or hyperparameters).

Any discussion is welcome!
