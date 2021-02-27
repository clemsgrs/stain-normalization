import os
import re
import json
import random
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from pathlib import Path
from shutil import rmtree
import inspect
import collections

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def open_config_file(filepath):
    with open(filepath) as jsonfile:
        pdict = json.load(jsonfile)
        params = AttrDict(pdict)
    return params

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_dir(path):
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError:
        rm = input('remove existing result dir? ')
        if rm:
            rmtree(path.as_posix())

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def print_current_errors(epoch, it, errors, t, save_log=False, log_path=''):
    message = f'(epoch: {epoch}, iters: {it}, time: {t:.3f})'
    for k, v in errors.items():
        message = f'{message} {k}: {v:.3f}'
    print(message)
    if save_log:
        with open(log_path, 'a') as log_file:
            log_file.write(f'{message}\n')

class ImagePool():
    
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        
        if self.pool_size == 0:
            return Variable(images)
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return_images = Variable(torch.cat(return_images, 0))
        
        return return_images