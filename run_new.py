import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import utils
import trainer
import argparse
from dataclasses import dataclass, field, fields, MISSING
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Config:
    num_layers: int = 25
    batch_size: int = 2
    hidden_dim: int = 32
    epochs: int = 300
    seed: int = 42
    opt: str = 'adam'
    opt_scheduler: str = 'step'
    opt_decay_step: int = 50
    opt_decay_rate: float = 0.5
    weight_decay: float = 5e-4
    lr: float = 0.002
    train_size: int = 107
    test_size: int = 28
    test_interval: int = 1
    plot_interval: int = 50
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    dataset: str = 'dataset5_135samples_withload_outerindex_107_28.pt'
    shuffle: bool = False
    save_best_model: bool = False
    checkpoint_dir: str = 'checkpoint/'

def create_parser(cls):
    """Dynamically create an ArgumentParser from a dataclass."""
    parser = argparse.ArgumentParser(description="Auto-generate parser from dataclass")
    
    for f in fields(cls):
        arg_name = f"--{f.name}"  # Always treat fields as optional

        kwargs = {}
        # Ensure default values are correctly handled
        if f.default is not MISSING:
            kwargs["default"] = f.default
        elif f.default_factory is not MISSING:
            kwargs["default"] = f.default_factory()
        else:
            kwargs["required"] = True  # Only mark required if no default exists

        # Handle booleans separately with store_true/store_false
        if f.type is bool:
            if f.default is False:
                kwargs.pop("default", None)
                kwargs["action"] = "store_true"
            else:
                kwargs.pop("default", None)
                kwargs["action"] = "store_false"
        else:
            kwargs["type"] = f.type

        parser.add_argument(arg_name, **kwargs)
    
    return parser

  

def main():
    parser = create_parser(Config)
    args = parser.parse_args()
    config = Config(**vars(args))
    print(config.device)
    dataset_full_timesteps = torch.load(config.dataset)
    dataset = dataset_full_timesteps[:args.train_size+args.test_size]
    if config.shuffle:
        random.shuffle(dataset)
    stats_list = utils.get_stats(dataset)
    trainer.train(dataset, stats_list, config)

if __name__ == "__main__":
    main()
