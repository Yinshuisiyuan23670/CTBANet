1.CTBANet
CTBANet: Convolution Transformers and Bidirectional Attention for Medical Image Segmentation

2.Prepare data
CTBANet/CTBA/datasets/README.md

3.Environment
python 3.10.12
pytorch 2.0.1

4.Train/Test
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse 
```

```bash
python test.py --dataset Synapse 
```

Reference 
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [DA-TransUnet](https://github.com/SUN-1024/DA-TransUnet)
