from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np

from core.inference import get_max_preds
import pdb

def get_bboxes(cam, cam_thr=0.5):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    #pdb.set_trace()
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox  #, thr_gray_heatmap, len(contours)

def cal_iou(box1, box2, method='iou'):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    if method == 'iog':
        iou_val = i_area / (box2_area)
    elif method == 'iob':
        iou_val = i_area / (box1_area)
    else:
        iou_val = i_area / (box1_area + box2_area - i_area)
    return iou_val

def accuracy(cfg, output_dir, input, output, target, name, bbox_label, is_Train = True, thr=0.5,):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    
    cnt = output.shape[0]
    acc = np.zeros(cnt)
    tar_acc = np.zeros(cnt)
    pd_bbox = []

    for b in range(cnt):
        out_b = output[b,0, :, :] 
        
        out_b = cv2.resize(out_b , (256, 256))
        
        cam_min, cam_max = out_b.min(), out_b.max()
        out_b = (out_b - cam_min) / (cam_max - cam_min)
        out_bbox = np.array(get_bboxes(out_b, cam_thr=thr) )
        pd_bbox.append(out_bbox)
        #pdb.set_trace()
        tar_b = target[b,0, :, :] 
        tar_b = cv2.resize(tar_b , (256, 256))
        tar_bbox = np.array(get_bboxes(tar_b, cam_thr=thr))
        iou = cal_iou(out_bbox, tar_bbox)
        tar_acc[b] = iou
        #pdb.set_trace()
        if not is_Train:
            max_iou = 0
            max_iou_tar = 0
            gt_bbox = bbox_label[b].strip().split(' ')
            gt_bbox = list(map(float, gt_bbox))
            gt_box_cnt = len(gt_bbox) // 4
            for i in range(gt_box_cnt):
                gt_box = gt_bbox[i * 4:(i + 1) * 4]
                iou_i = cal_iou(out_bbox, gt_box)
                iou = cal_iou(tar_bbox, gt_box)
                if iou_i > max_iou:
                    max_iou = iou_i
                if iou > max_iou_tar:
                    max_iou_tar = iou
            acc[b] = max_iou
            tar_acc[b] = max_iou_tar
            if cfg.TEST.SAVE_BOXED_IMAGE:
                inp_b = input[b]
                save_grid_img(inp_b,tar_b,out_b,name[b],output_dir,gt_bbox,out_bbox,max_iou,256,cfg.TEST.EVAL_SIZE)
    return acc, tar_acc, cnt, pd_bbox

def save_grid_img(input, target, output, name, savedir ,gt_bbox ,out_bbox,iou, size = 256, eval_size=64):
    #image = tensor2image(input,size)
    image_path = os.path.join('/home/xujy/Work/Dataset/CUB_200_2011','images',name)
    oriimage = cv2.imread(image_path)
    image = cv2.resize(oriimage,(size,size))
    savename = savedir+'result/'
    if not os.path.exists(savename): 
        os.mkdir(savename)
    savename = savename + name.split('/')[-1]
    grid_img = np.zeros((size*2,size*2,3),dtype=np.uint8)

    #change heat_target to pth
    cam_ori = torch.load(os.path.join('/home/xujy/Work/Dataset/CUB_200_2011/DinoCAM',name[:-4]+'.pth'))
    cam = torch.mean(cam_ori[:3,:], dim=0, keepdim=False)
    #heat_target = cv2.applyColorMap(resize_cam(target,size), cv2.COLORMAP_JET)
    heat_target = cv2.applyColorMap(resize_cam(cam.numpy(),size), cv2.COLORMAP_JET)
    heat_output = cv2.applyColorMap(resize_cam(output,size), cv2.COLORMAP_JET)
    blend = image * 0.3 + heat_output * 0.7
    #pdb.set_trace()
    boxed_image = draw_bbox(blend, iou, (size//eval_size)*np.array(gt_bbox).reshape(-1,4).astype(np.int),  (size//64)*np.array(out_bbox), iou, False)
    grid_img[:size,:size] = image
    grid_img[:size,size:] = boxed_image
    grid_img[size:,:size] = heat_target
    grid_img[size:,size:] = heat_output
    cv2.imwrite(savename,grid_img)
    #pdb.set_trace()
    return
