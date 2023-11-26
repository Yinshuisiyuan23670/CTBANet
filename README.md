# CTBANet
CTBANet: Convolution Transformers and Bidirectional Attention for Medical Image Segmentation

### 2.Prepare data
Please refer to /home/featurize/work/4DA/DA-TransUnet/DA-TransUNet/DA-TransUNet/datasets/README.md

### 3.Environment
Please prepare an environment with python=3.7(conda create -n envir python=3.7.12), and then use the command "pip install -r requirements.txt" for the dependencies.

### 4.Train/Test
Run the train script on synapse dataset. The batch size can be reduced to 12 or 16 to save memory(please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse 
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse 
```

## Reference 
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [DA-TransUnet](https://github.com/SUN-1024/DA-TransUnet)
