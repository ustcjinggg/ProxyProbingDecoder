import shutil
import os
import sys
import errno
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn

import math

def rm(path):
  try:
    shutil.rmtree(path)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise


def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


class AttrDict(dict):
    """
    Subclass dict and define getter-setter.
    This behaves as both dict and obj.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


class Logger(object):
    def __init__(self,filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename,'a')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def fix_seed_all(cfg):
    # fix sedd
    fix_random_seed(cfg.BASIC.SEED)
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE


def backup_codes(root_dir, res_dir, backup_list):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir) # delete
    os.makedirs(res_dir)
    for name in backup_list:
        shutil.copytree(os.path.join(root_dir, name), os.path.join(res_dir, name))
    print('codes backup at {}'.format(os.path.join(res_dir, name)))

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)