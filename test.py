import os
import sys
import time
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CreateDataLoader
from models import create_model
from utils import open_config_file

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="default.json", metavar='N', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)

params.gpu_ids = [params.gpu_ids]
# set gpu ids
if len(params.gpu_ids) > 0:
    torch.cuda.set_device(params.gpu_ids[0])

params.nThreads = 1  # test code only supports nThreads = 1
params.batchSize = 1  # test code only supports batchSize = 1
params.serial_batches = True  # no shuffle
params.no_flip = True  # no flip

###

data_loader = CreateDataLoader(params)
dataset = data_loader.dataset
dataset_size = len(dataset)
print(f'#testing images = {dataset_size}')

model = create_model(params)
start_time = time.time()

with tqdm(
    data_loader,
    desc=(f'Test'),
    unit=' imgs',
    ncols=80,
    unit_scale=params.batchSize) as t:

    for i, data in enumerate(t):

        if params.how_many:
            print(f'how_many: {params.how_many}')
            if i >= params.how_many:
                break
        
        model.set_input(data)
        model.test()
        
        # OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
        # ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        frameid = img_path[0].split('/')[-1].replace('.png', '')

        fig, axarr = plt.subplots(1,3, figsize=(12,5))
        axarr[0].imshow(visuals['real_A'])
        axarr[0].set_title('real Aperio', y=-0.1)
        axarr[0].axis('off')
        axarr[1].imshow(visuals['fake_B'])
        axarr[1].set_title('fake Hamamatsu', y=-0.1)
        axarr[1].axis('off')
        axarr[2].imshow(visuals['rec_A'])
        axarr[2].set_title('rec. Aperio', y=-0.1)
        axarr[2].axis('off')
        fig.suptitle(f'Aperio to Hamamatsu ({frameid})', fontsize=16)
        fig.tight_layout()
        plt.savefig(os.path.join(params.results_dir, f'{frameid}.pdf'))

elapsed = (time.time() - start_time)
print(f'--- {round((elapsed),2)} seconds ---')