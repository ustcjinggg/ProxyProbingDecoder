# ----------------------------------------------------------------------------------------------------------
# PPD-XJY
# Jan,2022
# ----------------------------------------------------------------------------------------------------------
import os
import cv2
import sys
import pdb
import xlwt
import time
import random
import datetime
import pprint
import numpy as np

import _init_paths
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader,\
    AverageMeter, accuracy, list2acc, adjust_learning_rate
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_dino_loc
from cams_deit import tensor2image

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import models.dino as vits
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from timm.optim import create_optimizer
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main():
    args = update_config()
    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join('./result/', cfg.DATA.DATASET, 'TIME{}_THR{}_BS{}_MASK{}_GAMMA_{}'.format(
        cfg.BASIC.TIME, cfg.MODEL.CAM_THR, cfg.TEST.BATCH_SIZE, cfg.DATA.MASK_SIZE, cfg.BASIC.GAMMA))
    cfg.BASIC.ROOT_DIR = './'
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log'); mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt'); mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)
    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)
    
    #Data_Loader
    train_loader, val_loader = creat_data_loader(cfg, cfg.DATA.DATADIR)
    
    # build model using pretrained DINO-S
    model = vits.__dict__['vit_small'](patch_size=cfg.BASIC.PATCH_SIZE, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Using Device: {}'.format(device))
    model.to(device)
    state_dict = torch.load('./models/dino_deitsmall16_pretrain.zip', map_location="cpu")
    if 'teacher' in state_dict:
        print(f"Take key teacher in provided checkpoint dict")
        state_dict = state_dict['teacher']
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded Encoder with msg: {}'.format(msg))
    
    #Load classifier
    embed_dim = model.embed_dim * (cfg.BASIC.N_LAST + int(cfg.BASIC.AVGPOOL))
    linear_classifier = vits.LinearClassifier(embed_dim, num_labels=cfg.DATA.NUM_CLASSES)
    if cfg.DATA.DATASET == 'CUB':
        fc_dict = torch.load('./models/cub_classifier.pth.tar', map_location="cpu")
    else:
        fc_dict = torch.load('./models/dino_deitsmall16_linearweights.pth', map_location="cpu")
    fc_dict = fc_dict['state_dict']
    fc_dict = {k.replace("module.", ""): v for k, v in fc_dict.items()}
    linear_classifier = linear_classifier.cuda()
    msg = linear_classifier.load_state_dict(fc_dict, strict=False)
    print('Loaded Classifier with msg: {}'.format(msg))
    
    #Load regressor
    decov_regressor = vits.DeconvNet()
    if cfg.DATA.DATASET == 'CUB':
        rg_dict = torch.load('./models/cub_localizer.pth')
    else:
        rg_dict = torch.load('./models/localizer.pth')
    decov_regressor = decov_regressor.cuda()
    msg = decov_regressor.load_state_dict(rg_dict)
    print('Loaded Regressor with msg: {}'.format(msg))

    update_val_step = 0
    update_val_step, loc_gt_known, iou_all = \
        val_loc_one_epoch(val_loader,train_loader, model, linear_classifier, decov_regressor, cfg.BASIC.N_LAST, cfg.BASIC.AVGPOOL, device, writer, cfg, update_val_step)
    print('Loc_gt:{0:.3f}\tAvg_iou:{1:.3f}\n'.format( loc_gt_known,iou_all))
    print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

def add_rst(sheet,img_name,bbox,iou_gt,cls_logits,target,best_thre):
    for idx,name in enumerate(img_name):
        sheet.write(idx,0,name)
        sheet.write(idx,1,bbox[idx][0])#'{:.4f}'.format
        sheet.write(idx,2,bbox[idx][1])
        sheet.write(idx,3,bbox[idx][2])
        sheet.write(idx,4,bbox[idx][3])
        sheet.write(idx,5,'{:.3f}'.format(iou_gt[idx]))
        loss = nn.CrossEntropyLoss()(cls_logits[idx].unsqueeze(0), target[idx].unsqueeze(0)).item()
        sheet.write(idx,6,'{:.5f}'.format(loss))
        sheet.write(idx,7,'{:.2f}'.format(best_thre[idx]))
    return

def val_loc_one_epoch(val_loader, train_loader, model, linear_classifier,decov_regressor, n_last, avgpool, device, writer, cfg, update_val_step):
    
    loc_gt_known = []
    iou_gt = []
    cls_top1 = []
    cls_top5 = []
    loc_top1 = []
    loc_top5 = []
    top1_loc_right = []
    top1_loc_cls = []
    top1_loc_mins = []
    top1_loc_part = []
    top1_loc_more = []
    top1_loc_wrong = []
    
    if cfg.DATA.TRAIN_LOADER:
        loader = train_loader
    else:
        loader = val_loader
    with torch.no_grad():
        model.eval()
        linear_classifier.eval()
        for i, one_loader in enumerate(loader):
            #if(i<=5150):
            #    continue
            start = time.time()
            if(len(one_loader) == 5):
                (input, cls_image, target, bbox, image_names) = one_loader
            else:
                input = one_loader[0]
                cls_image = one_loader[1]
                target = one_loader[2]
                image_names = one_loader[3]
                bs = input.shape[0]
                ptype = '1 1 479 479'
                bbox = [ptype for i in range(bs)]
            # update iteration steps
            update_val_step += 1
            bs = input.shape[0]
            target = target.to(device)
            input = input.to(device)
            #pdb.set_trace()
            cls_image = cls_image.to(device)
            
            #get cls logits
            intermediate_output = model.get_intermediate_layers(cls_image, n_last)
            output = [x[:, 0] for x in intermediate_output]
            feat = [x[:, 1:] for x in intermediate_output]
            if avgpool:
                output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            output = torch.cat(output, dim=-1) 
            feat = torch.cat(feat, dim=-1)                 
            cls_logits = linear_classifier(output) 
            
            _, topk_idx = cls_logits.topk(5, 1, True, True)
            topk_idx = torch.cat([topk_idx,target.unsqueeze(1)],dim=1)
            topk_idx = topk_idx.tolist()
            
            #get loc
            #pdb.set_trace()
            patch_embed = model.get_intermediate_layers(input,1)[0]
            hw = int((patch_embed.shape[1]-1)**0.5)
            patch_embed = patch_embed[:,1:,:].permute([0, 2, 1]).contiguous()
            patch_embed = patch_embed.reshape(bs,384,hw,hw)
            locmap = decov_regressor(torch.from_numpy(patch_embed.cpu().numpy()).cuda())
            att = []
            locmap = locmap.expand(bs,6,64,64)# 6 means that top5 and gt cls

            #else:
            all_tmp = []
            cam_total = torch.zeros(locmap.shape).to(device)
            
            gt_know, iou_all, pd_bbox, \
            cls_top1_b, cls_top5_b, loc_top1_b, loc_top5_b, \
              top1_loc_right_b, top1_loc_cls_b,top1_loc_mins_b, top1_loc_part_b, top1_loc_more_b, top1_loc_wrong_b, best_thre = \
                evaluate_dino_loc(input, target, bbox, cls_logits, att, locmap, cam_total, image_names, all_tmp, cfg, feat)
            
            loc_gt_known.extend(gt_know)
            iou_gt.extend(iou_all)
            
            cls_top1.extend(cls_top1_b)
            cls_top5.extend(cls_top5_b)
            loc_top1.extend(loc_top1_b)
            loc_top5.extend(loc_top5_b)
            top1_loc_right.extend(top1_loc_right_b)
            top1_loc_cls.extend(top1_loc_cls_b)
            top1_loc_mins.extend(top1_loc_mins_b)
            top1_loc_more.extend(top1_loc_more_b)
            top1_loc_part.extend(top1_loc_part_b)
            top1_loc_wrong.extend(top1_loc_wrong_b)

            #write into result
            workbook = xlwt.Workbook()
            sheet = workbook.add_sheet('0')
            add_rst(sheet,image_names,pd_bbox,iou_all,cls_logits,target,best_thre)
            save_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log/result_xls')
            workbook.save(save_dir + 'imgnet_bs{}.xls'.format(i))
            if i % cfg.BASIC.DISP_FREQ == 0 or i == len(loader)-1:
                print('Num{0}/{1}:\tLoc_gt:{2:.3f}\tAvg_IoU:{3:.3f}'.format(i,len(loader),list2acc(loc_gt_known),list2acc(iou_gt)))
                print('Cls@1:{0:.3f}\tCls@5:{1:.3f}\n'
                      'Loc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}'.format(
                    list2acc(cls_top1), list2acc(cls_top5),
                    list2acc(loc_top1), list2acc(loc_top5), list2acc(loc_gt_known)))
                print('Taking time for one loader: {:.1f}s\n'.format(time.time()-start))
    return update_val_step, list2acc(loc_gt_known), list2acc(iou_gt)
    

if __name__ == "__main__":
    main()
