import os 
import numpy as np
import pandas as pd
import argparse
import math
import json
import time
from copy import deepcopy
from typing import Union
from util.util import * 
from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho
from evaluator.eval_seeds import eval_seeds
from evaluator.eval_sample import eval_sampler

description = ""
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
parser.add_argument("method", help="synthesis method")
parser.add_argument("dataset", help="dataset to use")
parser.add_argument("device", help="device to use")
parser.add_argument("epsilon", type=float, help="privacy parameter")
parser.add_argument("--delta", type=float, default=1e-5, help="privacy parameter")
parser.add_argument("--num_preprocess", type=str, default='uniform_kbins')
parser.add_argument("--rare_threshold", type=float, default=0.002) # if 0 then 3sigma
parser.add_argument("--sample_device", help="device to synthesis, only used in some deep learning models", default=None)
parser.add_argument("--test", action="store_true")
parser.add_argument("--syn_test", action="store_true")
args = parser.parse_args()

if args.sample_device is None:
    args.sample_device = args.device

if args.method in ['rap', 'rap_syn'] and args.dataset in ['loan', 'higgs-small']:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"




def main(args):
    print(f'privacy setting: ({args.epsilon}, {args.delta})')
    parent_dir, data_path = make_exp_dir(args)
    time_record = {}

    # data preprocess
    if args.method == 'ddpm':
        total_rho = 1.0 # not used, set to 1.0 to calculate preprocesser_divide
        data_preporcesser = data_preporcesser_common(args)
        df, domain, preprocesser_divide  = data_preporcesser.load_data(data_path, 0)
    else:
        total_rho = cdp_rho(args.epsilon, args.delta)
        data_preporcesser = data_preporcesser_common(args)
        df, domain, preprocesser_divide  = data_preporcesser.load_data(data_path, total_rho) 
    

    # fitting model
    start_time = time.time()
    generator_dict = algo_method(args)(args, df=df, domain=domain, rho=(1-preprocesser_divide)*total_rho, parent_dir=parent_dir, preprocesser = data_preporcesser)
    end_time = time.time()
    time_record['model fitting time'] = end_time-start_time

    # evaluation
    eval_config = prepare_eval_config(args, parent_dir)

    if args.syn_test:
        test_config = deepcopy(eval_config)
        test_config['sample']['sample_num'] = 5000
        eval_sampler(args.method, test_config, args.sample_device, data_preporcesser, **generator_dict)
    
    if not args.test or not args.syn_test:
        eval_seeds(
            eval_config, 
            sampling_method = args.method,
            device = args.sample_device,
            preprocesser = data_preporcesser,
            time_record = time_record,
            **generator_dict
        ) 


if __name__ == "__main__":
    main(args)

