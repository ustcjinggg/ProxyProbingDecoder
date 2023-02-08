import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


from core.evaluate import accuracy
from core.inference import get_final_preds

from utils import fix_random_seed, backup_codes, rm

logger = logging.getLogger(__name__)

def prepare_env(cfg):
    # fix random seed
    fix_random_seed(cfg.BASIC.SEED)
    # cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  # Benchmark will impove the speed
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  #
    cudnn.enabled = cfg.CUDNN.ENABLE  # Enables benchmark mode in cudnn, to enable the inbuilt cudnn auto-tuner

    # backup codes
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'backup')
        rm(backup_dir)
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LIST)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_pd =  AverageMeter()

    # switch to train mode
    model.train()
    prfq = len(train_loader) // 8
    end = time.time()
    for i, (input, target, label, name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        input = input.cuda(non_blocking=True)
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        #target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, label)
            for output in outputs[1:]:
                loss += criterion(output, target, label)
        else:
            output = outputs
            loss = criterion(output, target, label)

        # loss = criterion(output, target, target_weight)
        #G_loss = GuassLoss(0.1,0.04).cuda()
        #gloss = G_loss(output)
        #loss += 0.001 * gloss
        #pdb.set_trace()
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % prfq == 0 or i+1 == len(train_loader):
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)
# \ 'TG-Acc {acc.val:.3f} ({acc.avg:.3f}) \t PD-Acc {pd.val:.3f} ({pd.avg:.3f})', acc=acc, pd=acc_pd
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            #prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            #save_debug_images(config, input, name, target, pred*4, output,prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_pd = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, 1, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    prfq = len(val_loader) // 6
    with torch.no_grad():
        end = time.time()
        for i, (input, target, cls_label, gt_bbox, name) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            target = target.cuda(non_blocking=True)

            loss = criterion(output, target, cls_label)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            pd_acc, tar_acc, cnt, pred_bbox = accuracy(config, output_dir,input.detach().cpu(),
                        output.detach().cpu().numpy(),target.detach().cpu().numpy(), name, gt_bbox, False, 0.5)
            
            pd_loc = len(pd_acc[pd_acc>=0.5])/len(pd_acc)
            #pdb.set_trace()
            acc_pd.update(pd_loc,cnt)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            idx += num_images

            if i % prfq == 0 or i+1 == len(val_loader):
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Predict-Acc {pd.val:.3f} ({pd.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, pd=acc_pd)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                #save_debug_images(config, input, meta, target, pred*4, output,
                #                  prefix)

    return acc_pd.avg


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class GuassLoss(nn.Module):
    def __init__(self, u = 0.1, sigma = 0.05):
        super(GuassLoss, self).__init__()
        self.u = u
        self.sigma = sigma

    def forward(self, x, ):
        loss =  torch.sum(torch.exp(-(x-self.u)**2/(2*self.sigma**2)))
        loss/= (x.shape[0]*x.shape[-1]*x.shape[-2])
        #pdb.set_trace()
        return loss