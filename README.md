# ProxyProbingDecoder
Repository for paper "Proxy Probing Decoder for Weakly Supervised Object Localization: A Baseline Investigation".

## Introduction
Our paper has been recieved by MM2022! You may get more detail in the paper.

## Requirement:
* torch >= 1.7.1
* numpy >= 1.19.5
* timm >= 0.3.2

## Data
Before you try this code, please download CUB dataset and change the datadir in yaml.

## Models
You can download the models here for evaluation:
https://drive.google.com/drive/folders/1UdSlfO0Iv-b0OYE8aF51zoO-jLgm1wbb?usp=sharing

## Start
python ./tools_cam/test_ppd.py --config ./configs/CUB/ppd_baseline.yaml
