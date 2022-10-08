# PMR-CNN

**PMR-CNN** is an open source implementation of our proposed methods Prototype Mixture R-CNN for Few-Shot Object Detection. This repo is built upon [Attention RPN](https://github.com/fanq15/FewX) and based on Detectron2. 

## Requirements
```
Linux with Python == 3.8.12
PyTorch == 1.5.0+cu10.1
Torchvision == 0.6.0+cuda10.1
Detectron2 == 0.3
CUDA 10.1
GCC == 7.3.0
```
* Install [PyTorch](https://pytorch.org/). You can choose the PyTorch and CUDA version according to your machine. Just make sure your PyTorch version matches the prebuilt Detectron2 version. 
```
# CUDA 10.1
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```
* Install [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) version according to your machine.
```
python -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

## Step1: Data Preparation
### COCO
We use [COCO 2017](https://cocodataset.org/#home) and keep the 5k images from minival set for evaluation and use the rest for training. We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.

`cd datasets/coco/`, change the `data_dir` and `root_path` in your data path.
```
cd PMR-CNN/datasets/coco/
python3 split_category.py
python3 voc_few_shot.py
python3 gen_base_support.py
python3 gen_novel_support_10_shot.py
```

## Step2: Base-Class training
We used ResNet101 as our backbone pretrained weights. Download the pretrained weights [R-101.pkl](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) and put it into the `PMR-CNN/configs/`. The `self.num_pro = 1` in base class training.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 fsod_train_net.py --num-gpus 8 --config-file configs/PMMv2/R_101.yaml
```

## Step3: Few-shot Fine-tuning
Fine-tune the base-training models on few-shot training data including only novel classes. Before training, we have to change the `self.num_pro = 2` in `pmrcnn/modeling/fsod/pmm_rcnn_v2`.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 --config-file configs/PMMv2/finetune_R_101.yaml
```

## Step4: Inference
Evaluation is conducted on the test set of COCO val2017.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 --config-file configs/PMMv2/finetune_R_101.yaml --eval-only MODEL.WEIGHTS ./output/pmmv2/R_101/model_final.pth
```
Here is the [model_final.pth](https://drive.google.com/drive/folders/1CpJV0P6uETSGDM_BYZ0mAPetDGZ97LzS?usp=sharing) and put it into PMR-CNN/output.

## Results on MS COCO 2017
We get the following result below:
| Shot |  AP  | AP50 | AP75 | APs | APm  | APl  |
|:----:|:----:|:----:|:----:|-----|------|------|
|   3  |  9.0 | 17.0 |  8.6 | 2.1 | 10.4 | 16.3 |
|   5  |  9.9 | 18.9 |  9.3 | 2.5 | 11.9 | 17.6 |
|  10  | 13.5 | 25.1 | 12.9 | 3.2 | 14.7 | 23.9 |
