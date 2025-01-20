
# Pytorch implementation of Adversarial Normalization: I Can visualize Everything (ICE)

We used foreground and background segmentation to evaluation ICE.

## Requirements
requirements.txt


## 1. How to train ICE with imagenet data:
```
cd imagenet_train
```
run
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 1234 --use_env main_ICE.py --resume https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --model deit_small_patch16_224 --batch-size 256 --data-path data_path/imagenet --output_dir ../output/ICE --epochs 3 --lr 0.0001
```




## 2. How to evaluate ICE with imagenet segmentation:
```
cd imagenet_segmentation
```
dataset
```
wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
```
run
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/imagenet_seg_eval_ICE.py --imagenet-seg-path ./gtsegs_ijcv.mat --checkpoint ../output/ICE/0_checkpoint.pth
```
 

## 3. How to evaluate ICE with jaccard similarity:

```
cd jaccard_similarity
```
run
```
python evaluate_segmentation_ICE.py   --model_name "dino_small"   --batch_size 256   --patch_size 16  --checkpoint ./output/ICE/0_checkpoint.pth
```

## References

imagenet_train code is based on DeiT(https://github.com/facebookresearch/deit) repository and TIMM library. 

imagenet_segmentation code is based on Intriguing Properties of Vision Transformers repository(https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers).

jaccard_similarity code is based on Transformer Interpretability Beyond Attention Visualization repository(https://github.com/hila-chefer/Transformer-Explainability). 

Thank you for the authors releasing their codes.

## Models
[Model checkpoint](https://drive.google.com/file/d/1zuuO40NPf-poWx-ncewj6MDV60n4LZiO/view?usp=sharing)

## Citing our paper
```
@InProceedings{Choi_2023_CVPR,
    author    = {Choi, Hoyoung and Jin, Seungwan and Han, Kyungsik},
    title     = {Adversarial Normalization: I Can Visualize Everything (ICE)},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {12115-12124}
}
```
