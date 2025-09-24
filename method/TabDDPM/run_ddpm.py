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
from util.rho_cdp import cdp_rho


def ddpm_main(args, df, domain, rho, parent_dir, **kwargs):
    # basic config
    if args.epsilon > 0 and not args.test:
        epsilon = args.epsilon
        delta = args.delta 
        total_rho = cdp_rho(epsilon, delta)
    else:
        epsilon = None 
        delta = None
        total_rho = None
    
    print(f'total privacy budget: ({epsilon}, {delta})')

    base_config_path = f'method/TabDDPM/exp/{args.dataset}/config.toml'
    base_config = load_config(base_config_path)

    data_info = load_json(f'data/{args.dataset}/info.json')
    dataset = make_dataset_from_df(
            df,
            T = Transformations(**base_config['train']['T']),
            y_num_classes = base_config['model_params']['num_classes'],
            is_y_cond = base_config['model_params']['is_y_cond'],
            task_type = data_info['task_type'],
            df_pub = kwargs.get('df_pub', None)
        )
    
    rho_used = kwargs.get('rho_used', None)
    rho_y = total_rho * 0.01 if total_rho is not None else None
    rho_used = rho_used + rho_y if rho_used is not None and rho_y is not None else rho_used or rho_y

    train_size = len(dataset.y['train'])
    base_config["parent_dir"] = parent_dir
    base_config['sample']['num_samples'] = train_size
    dump_config(base_config, f'{parent_dir}/config.toml')

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
    
    if rho_y is not None:
        _, empirical_class_dist = torch.unique(torch.from_numpy(dataset.y['train']), return_counts=True).float()
        empirical_class_dist += torch.randn(empirical_class_dist.shape) * np.sqrt(1/(2 * rho_y))
        empirical_class_dist = torch.clamp(empirical_class_dist, min=0)
    else:
        empirical_class_dist = None

    sampler = ddpm_sampler(
        diffusion=diffusion_model,
        num_numerical_features=base_config['num_numerical_features'],
        T_dict = base_config['train']['T'],
        dataset = dataset,
        model_params = base_config['model_params'],
        empirical_class_dist = empirical_class_dist
    )

    return {"ddpm_generator": sampler} 
    
