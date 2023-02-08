import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import pdb
import cv2
import logging
import h5py

logger = logging.getLogger(__name__)

def get_transforms(cfg):
    train_transform = transforms.Compose([
        transforms.Resize(cfg.MODEL.IMAGE_SIZE),
        #transforms.RandomCrop((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
    ])
    loc_transform = transforms.Compose([
        transforms.Resize(cfg.MODEL.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
    ])
    loc_detail = np.zeros(900)
    val_transform = transforms.Compose([
        transforms.Resize(cfg.MODEL.IMAGE_SIZE, interpolation=3),
        #transforms.CenterCrop(cfg.DATA.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD),
    ])
    test_tencrops_transform = transforms.Compose([
        transforms.Resize(cfg.MODEL.IMAGE_SIZE),
        #transforms.TenCrop(cfg.DATA.CROP_SIZE),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
             (transforms.ToTensor()(crop)) for crop in crops])),
    ])
    return train_transform, val_transform, loc_transform, test_tencrops_transform


class CUBFEATDataset(Dataset):
    def __init__(self, root, cfg, is_train):
        self.root = root
        self.cfg = cfg
        self.is_train = is_train
        self.resize_size =  cfg.DATA.RESIZE_SIZE
        self.loc_size = cfg.DATA.LOC_SIZE
        self.flip = True
        self.color_rgb = False
        self.heatmap_size = np.array([64,64])
        self.target_dir = os.path.join(self.root, 'DinoCAM2')

        self.image_list = self.remove_1st_column(open(
            os.path.join(root, 'images.txt'), 'r').readlines())
        self.label_list = self.remove_1st_column(open(
            os.path.join(root, 'image_class_labels.txt'), 'r').readlines())
        self.split_list = self.remove_1st_column(open(
            os.path.join(root, 'train_test_split.txt'), 'r').readlines())
        self.bbox_list = self.remove_1st_column(open(
            os.path.join(root, 'bounding_boxes.txt'), 'r').readlines())

        if is_train:
            self.index_list = self.get_index(self.split_list, '1')
        else:
            self.index_list = self.get_index(self.split_list, '0')
        
        if self.is_train:
            self.image_dir = os.path.join(self.root, 'train')
            feat_path = os.path.join(self.root, 'DinoFeat2', 'feat_train.h5')
            with h5py.File(feat_path,'r') as f:
                data = f['dataset']
                self.feat_numpy = data[:]
            with h5py.File(os.path.join(self.target_dir,'cam_train.h5'),'r') as f:
                data = f['dataset']
                self.target_numpy = data[:]
        else:
            feat_path = os.path.join(self.root, 'DinoFeat2', 'feat_val.h5')
            self.image_dir = os.path.join(self.root, 'val')
            with h5py.File(feat_path,'r') as f:
                data = f['dataset']
                self.feat_numpy = data[:]
            with h5py.File(os.path.join(self.target_dir,'cam_val.h5'),'r') as f:
                data = f['dataset']
                self.target_numpy = data[:]

        

    # For multiple bbox
    def __getitem__(self, idx):
        name = self.image_list[self.index_list[idx]]
        
        feat = torch.from_numpy(self.feat_numpy[idx])
        #print(feat.shape)
        feat = feat[:,:,384:]
        feat = feat.permute([2, 0, 1])
        feat = feat.contiguous()
        label = int(self.label_list[self.index_list[idx]])-1

        target = self.generate_target(idx)
        
        target = torch.from_numpy(target).unsqueeze(0)
        
        input  = feat
        if self.is_train:   
            #input = self.train_transform(input)
            return input, target, label, name

        else:
            image_path = os.path.join(self.root, 'images', name)
            image = Image.open(image_path).convert('RGB')
            image_size = list(image.size)
            #input = self.loc_transform(input)
            bbox = self.bbox_list[self.index_list[idx]]
            bbox = [int(float(value)) for value in bbox]
            [x, y, bbox_width, bbox_height] = bbox

            resize_size = 256
            shift_size = 0
            [image_width, image_height] = image_size
            left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
            left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))
            right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, resize_size - 1))
            right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, resize_size - 1))

            gt_bbox = np.array([left_bottom_x, left_bottom_y, right_top_x, right_top_y]).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))
            return input, target, label, gt_bbox, name
    
    def get_index(self, list, value):
        index = []
        for i in range(len(list)):
            if list[i] == value:
                index.append(i)
        return index
    
    def remove_1st_column(self, input_list):
        output_list = []
        for i in range(len(input_list)):
            if len(input_list[i][:-1].split(' '))==2:
                output_list.append(input_list[i][:-1].split(' ')[1])
            else:
                output_list.append(input_list[i][:-1].split(' ')[1:])
        return output_list

    def __len__(self):
        return len(self.index_list)

    def generate_target(self,idx):
        cam_ori = torch.from_numpy(self.target_numpy[idx])#torch.load(os.path.join(self.target_dir, name[:-4] +'.pth'))
        cam = torch.mean(cam_ori[:3,:], dim=0, keepdim=False)
        #print(cam.max())
        cam_np = cv2.resize(cam.numpy() , (self.heatmap_size[0], self.heatmap_size[1]))
        cam_min, cam_max = cam_np.min(), cam_np.max()
        target = (cam_np - cam_min) / (cam_max - cam_min)
        #try 0-1 distribution
        target[target>=0.05] = 1
        target[target<0.05] = 0

        return target

