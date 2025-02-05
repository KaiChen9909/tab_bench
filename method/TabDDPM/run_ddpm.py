import sys
target_path="./"
sys.path.append(target_path)

import os
import argparse
from copy import deepcopy
from method.TabDDPM.scripts.pretrain_and_finetune import finetune
from method.TabDDPM.scripts.sample import ddpm_sampler
from method.TabDDPM.data.dataset import * 
from method.TabDDPM.data.data_utils import * 



def ddpm_main(args, df, domain, rho, parent_dir, **kwargs):
    # basic config
    if args.epsilon > 0:
        epsilon = args.epsilon
        delta = args.delta 
    else:
        epsilon = None 
        delta = None
    
    print(f'training privacy budget: ({epsilon},{delta})')

    base_config_path = f'method/TabDDPM/exp/{args.dataset}/config.toml'
    base_config = load_config(base_config_path)

    data_info = load_json(f'data/{args.dataset}/info.json')
    dataset = make_dataset_from_df(
            df,
            T = Transformations(**base_config['train']['T']),
            y_num_classes = base_config['model_params']['num_classes'],
            is_y_cond = base_config['model_params']['is_y_cond'],
            task_type = data_info['task_type']
        )
    
    train_size = len(dataset.y['train'])
    base_config["parent_dir"] = parent_dir
    base_config['sample']['num_samples'] = train_size
    dump_config(base_config, f'{parent_dir}/config.toml')

    rho_used = kwargs.get('rho_used', None)

    # fit the diffusion model
    diffusion_model = finetune(
            **base_config['train']['main'],
            **base_config['diffusion_params'],
            parent_dir=base_config['parent_dir'],
            dataset = dataset,
            model_type=base_config['model_type'],
            model_params=base_config['model_params'],
            model_path = None,
            T_dict=base_config['train']['T'],
            num_numerical_features=base_config['num_numerical_features'],
            device=args.device,
            dp_epsilon = epsilon,
            dp_delta = delta,
            rho_used = rho_used,
            report_every = args.test
        ) 

    sampler = ddpm_sampler(
        diffusion=diffusion_model,
        num_numerical_features=base_config['num_numerical_features'],
        T_dict = base_config['train']['T'],
        dataset = dataset,
        model_params = base_config['model_params']
    )

    return {"ddpm_generator": sampler}
    
