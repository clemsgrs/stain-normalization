import os
import sys
import time
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import make_dataset, get_transform
from models import create_model
from utils import open_config_file

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config/inference_config.json", metavar='N', help='config file')
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
transform = get_transform(params)
start_time = time.time()

for i, img_path in enumerate(inference_images):

    if params.how_many:
        if i >= params.how_many:
            print(f'how_many: {params.how_many}')
            break
    
    frameid = img_path.split('/')[-1].replace('.png', '')
    target_frameid = f'H{frameid[1:]}'
    target_path = os.path.join(params.target_dir, f'{target_frameid}.png')

    source_img = Image.open(img_path).convert('RGB')
    target_img = Image.open(target_path).convert('RGB')
    data = transform(source_img).unsqueeze(0)
    target = transform(target_img).unsqueeze(0)

    model.set_input({'A': data, 'B': target, 'A_path': img_path, 'B_path': target_path})
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
    axarr[2].imshow(visuals['real_B'])
    axarr[2].set_title('true Hamamatsu', y=-0.1)
    axarr[2].axis('off')
    fig.suptitle(f'Aperio to Hamamatsu ({frameid})', fontsize=16)
    fig.tight_layout()
    plt.savefig(os.path.join(params.results_dir, f'{frameid}.pdf'))

elapsed = (time.time() - start_time)
print(f'--- {round((elapsed),2)} seconds ---')