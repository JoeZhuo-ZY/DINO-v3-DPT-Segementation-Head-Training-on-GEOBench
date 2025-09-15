### Minimal reproduction of DINOv3 (ViT) encoder + DPT head for 6 segmentation tasks on GeoBench.
The official training code for SegHead was not released; this repo provides a training loop, a DPT decoder head, and a GeoBench wrapper that supports single-task and multi-task runs.

The DPT head code is from a Meta [repo](https://github.com/facebookresearch/HighResCanopyHeight/blob/main/models/dpt_head.py#L245).
The GeoBench dataset wrapper is from DOFA [repo](https://github.com/xiong-zhitong/DOFA-pytorch/blob/geofm/src/datasets/geobench_wrapper.py).

