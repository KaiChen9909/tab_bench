import sys
target_path="./"
sys.path.append(target_path)

import argparse
from DP_MERF.single_generator_priv_all import * 
from TabDDPM.data.data_utils import dump_json

import warnings
warnings.filterwarnings('ignore')

import os

args=argparse.ArgumentParser()
args.add_argument("dataset", default="covtype")
args.add_argument("epochs", type=int)
args.add_argument("batch", type=float, default=0.1)
args.add_argument("num_features", type = int, default=1000)
args.add_argument("device", default='cuda:0')
args.add_argument("--dp_epsilon", type=float, default=None)
args.add_argument("--num_preprocess", type=str, default='privtree')
args.add_argument("--rare_threshold", type=float, default=0.005)

# args.add_argument("--undersample", type=float, default=1)
# args.add_argument('--classifiers', nargs='+', type=int, help='list of integers', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# args.add_argument("--data_type", default='generated') #both, real, generated 
arguments=args.parse_args()
print("arg", arguments)

device = arguments.device
parent_dir = f'DP_MERF/exp/{arguments.dataset}/{arguments.dp_epsilon}_dp_merf_{arguments.num_preprocess}_{arguments.rare_threshold}'


if __name__ == '__main__':

    os.makedirs(parent_dir, exist_ok=True)

    dp_epsilon = arguments.dp_epsilon
    is_priv_arg = (dp_epsilon is not None)

    merf_main(
        arguments.dataset, 
        dp_epsilon, 
        device,
        n_features_arg = arguments.num_features, 
        mini_batch_size_arg = arguments.batch, 
        how_many_epochs_arg = arguments.epochs, 
        is_priv_arg = is_priv_arg, 
        parent_dir = parent_dir,
        seed_number = 0,
        num_prep = arguments.num_preprocess,
        rare_threshold = arguments.rare_threshold
    )