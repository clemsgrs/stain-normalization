import os
import torch
import argparse
import time
from tqdm import tqdm

from my_dataset import CreateDataLoader
from my_models import create_model
from my_utils import open_config_file, print_current_errors, mkdirs, mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="default.json", metavar='N', help='config file')
args = parser.parse_args()
params = open_config_file(args.config)


params.gpu_ids = [params.gpu_ids]
# set gpu ids
if len(params.gpu_ids) > 0:
    torch.cuda.set_device(params.gpu_ids[0])

args = vars(params)

print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# save to the disk
expr_dir = os.path.join(params.checkpoints_dir, params.name)
mkdir(expr_dir)
file_name = os.path.join(expr_dir, 'params.txt')
with open(file_name, 'wt') as params_file:
    params_file.write('------------ Options -------------\n')
    for k, v in sorted(args.items()):
        params_file.write('%s: %s\n' % (str(k), str(v)))
    params_file.write('-------------- End ----------------\n')

###

data_loader = CreateDataLoader(params)
dataset = data_loader.load_data()
# dataset = data_loader.dataset
dataset_size = len(data_loader)
print(f'#training images = {dataset_size}')

model = create_model(params)
total_steps = 0

for epoch in range(params.epoch_count, params.niter + params.niter_decay + 1):
    
    epoch_start_time = time.time()
    epoch_iter = 0
    
    with tqdm(data_loader,
              desc=(f'Train - Epoch: {epoch}'),
              unit=' imgs',
              ncols=80,
              unit_scale=params.batchSize) as t:

        for i, data in enumerate(t):
            
            iter_start_time = time.time()
            total_steps += params.batchSize
            epoch_iter += params.batchSize
            model.set_input(data)

            # combined forward + backward pass
            model.optimize_parameters()

        if epoch % params.save_epoch_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - epoch_start_time)
            log_filepath = os.path.join(expr_dir, 'log', 'loss_log.txt')
            print_current_errors(epoch, epoch_iter, errors, t, params.save_log, log_filepath)
            print(f'\saving the model at the end of epoch {epoch}, iters {total_steps}')
            model.save('latest')
            # model.save(epoch)

        print(f'End of epoch {epoch} / {params.niter + params.niter_decay} \t Time Taken: {time.time() - epoch_start_time} sec')
        model.update_learning_rate()