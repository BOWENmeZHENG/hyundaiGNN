import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import random
import matplotlib.pyplot as plt
import numpy as np
import utils
import trainer
import warnings
warnings.filterwarnings('ignore')

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

for args in [
        {'num_layers': 25,
         'batch_size': 2,
         'hidden_dim': 32,
         'epochs': 200,
         'seed':42,
         'opt': 'adam',
         'opt_scheduler': 'step',
         'opt_decay_step': 50,
         'opt_decay_rate': 0.5,
         'opt_restart': 0,
         'weight_decay': 5e-4,
         'lr': 0.002,
         'train_size': 107,
         'test_size': 28,
         'test_interval': 1,
         'plot_interval': 40,
         'device':'cuda',
         'shuffle': True,
         'save_best_model': False,
         'checkpoint_dir': "checkpoint/",
         },
    ]:
        args = objectview(args)
dataset_full_timesteps = torch.load("dataset5_135samples_withload_outerindex_107_28.pt")
dataset = dataset_full_timesteps[:args.train_size+args.test_size]
if args.shuffle:
    random.shuffle(dataset)

stats_list = utils.get_stats(dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
print(device)

test_losses, test_r2, losses, best_model, best_test_loss, test_loader = trainer.train(dataset, device, stats_list, args)

