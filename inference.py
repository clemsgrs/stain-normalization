import os
import sys
import time
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import make_dataset
from models import create_model
from utils import open_config_file

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="inference_config.json", metavar='N', help='config file')
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

inference_images = make_dataset(params.source_dir)
num_inference_images = len(inference_images)
print(f'#inference images = {num_inference_images}')

model = create_model(params)
start_time = time.time()

for i, img_subpath in enumerate(inference_images):

    if params.how_many:
        print(f'how_many: {params.how_many}')
        if i >= params.how_many:
            break
    
    frameid = img_subpath[0].replace('.png', '')
    img_path = os.path.join(params.source_dir, img_subpath)
    target_path = os.path.join(params.target_dir, img_subpath)

    data = Image.open(img_path).convert('RGB')
    target = Image.open(target_path).convert('RGB')

    model.set_input(data)
    model.test()
    
    # OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
    # ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
    visuals = model.get_current_visuals()
    # img_path = model.get_image_paths()

    fig, axarr = plt.subplots(1,3, figsize=(12,5))
    axarr[0].imshow(visuals['real_A'])
    axarr[0].set_title('real Aperio', y=-0.1)
    axarr[0].axis('off')
    axarr[1].imshow(visuals['fake_B'])
    axarr[1].set_title('fake Hamamatsu', y=-0.1)
    axarr[1].axis('off')
    axarr[2].imshow(target)
    axarr[2].set_title('true Hamamatsu', y=-0.1)
    axarr[2].axis('off')
    fig.suptitle(f'Aperio to Hamamatsu ({frameid})', fontsize=16)
    fig.tight_layout()
    plt.savefig(os.path.join(params.results_dir, f'{frameid}.pdf'))

elapsed = (time.time() - start_time)
print(f'--- {round((elapsed),2)} seconds ---')