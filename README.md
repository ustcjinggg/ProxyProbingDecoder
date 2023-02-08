# ProxyProbingDecoder
Repository for paper "Proxy Probing Decoder for Weakly Supervised Object Localization: A Baseline Investigation".

## Introduction
Our paper has been recieved by MM2022! You may get more detail in the paper. This repo is build based on TS-CAM, DINO and Simple-baseline.

## Requirement:
* torch >= 1.7.1
* numpy >= 1.19.5
* timm >= 0.3.2

## Data
Before you try this code, please download CUB dataset and change the datadir in yaml. 
Your datadir should look like this: 
CUB_200_2011 
  --images 
  --DinoFeat2 
  --DinoCAM2 
  --list 
We use fixed Dino-S model to provide features and pseudo labels in DinoFeat2 and DinoCAM2. You can download from:



## Models
You can download the models here for evaluation:
https://drive.google.com/drive/folders/1UdSlfO0Iv-b0OYE8aF51zoO-jLgm1wbb?usp=sharing

## Start
python ./tools_cam/test_ppd.py --config ./configs/CUB/ppd_baseline.yaml
python ./tools_cam/train_ppd.py --config ./configs/CUB/train_ppd_baseline.yaml
