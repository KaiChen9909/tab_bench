# This file is not used in experiment

import tomli
import shutil
import os
import argparse
from pretrain_and_finetune import pretrain, finetune
from sample import presample,sample
from eval_mlp import train_mlp
from eval_simple import train_simple
from eval_catboost import train_catboost
from data.data_utils import load_config
import pandas as pd
import matplotlib.pyplot as plt
import torch


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--presample', action='store_true',  default=False)
    parser.add_argument('--preeval', action='store_true',  default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)
    parser.add_argument('--dp_epsilon', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrain_model_path', type=str, default=None)

    args = parser.parse_args()
    raw_config = load_config(args.config)
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda:0')
    
    #timer = zero.Timer()
    #timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.pretrain:
        pretrain(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val
        )
    if args.presample:
        presample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'pretrain_model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.preeval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                model_step = 'pretrain'
            )
        elif raw_config['eval']['type']['eval_model'] == 'mlp':
            train_mlp(
                parent_dir=raw_config['parent_dir'],
                data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                device=device,
                model_step = 'pretrain'
            )
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                parent_dir=raw_config['parent_dir'],
                data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                model_step = 'pretrain'
            )

    if args.finetune:
        finetune(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            model_path = args.pretrain_model_path,
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val,
            dp_epsilon = args.dp_epsilon
        )
    if args.sample:
        sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], f'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val,
            dp_epsilon = args.dp_epsilon
        )
    # save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                model_step = 'finetune'
            )
        elif raw_config['eval']['type']['eval_model'] == 'mlp':
            train_mlp(
                parent_dir=raw_config['parent_dir'],
                data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                device=device,
                model_step = 'finetune'
            )
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                parent_dir=raw_config['parent_dir'],
                data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                model_step = 'finetune'
            )


    # print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()