import torch
import argparse
import time

from my_dataset import CreateDataLoader
from my_models import create_model
from my_utils import open_config_file

parser = argparse.ArgumentParser()
parseradd_argument('--config', type=str, default="default.json", metavar='N', help='config file')
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
file_name = os.path.join(expr_dir, 'opt.txt')
with open(file_name, 'wt') as opt_file:
    opt_file.write('------------ Options -------------\n')
    for k, v in sorted(args.items()):
        opt_file.write('%s: %s\n' % (str(k), str(v)))
    opt_file.write('-------------- End ----------------\n')

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

    for i, data in enumerate(dataset):
        
        iter_start_time = time.time()
        total_steps += params.batchSize
        epoch_iter += params.batchSize
        model.set_input(data)

        # combined forward + backward pass
        model.optimize_parameters()

        if total_steps % params.display_freq == 0:
            save_result = total_steps % params.update_html_freq == 0

        if total_steps % params.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / params.batchSize

        if total_steps % params.save_latest_freq == 0:
            print(f'saving the latest model (epoch:{epoch}, total_steps:{total_steps})')
            model.save('latest')

    if epoch % params.save_epoch_freq == 0:
        print('saving the model at the end of epoch {epoch}, iters {total_steps}')
        model.save('latest')
        model.save(epoch)

    print(f'End of epoch {epoch} / {params.niter + params.niter_decay} \t Time Taken: {time.time() - epoch_start_time} sec')
    model.update_learning_rate()