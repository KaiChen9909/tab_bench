import sys
target_path="./"
sys.path.append(target_path)

import os
import argparse
from copy import deepcopy
from TabDDPM.scripts.pretrain_and_finetune import finetune
from TabDDPM.scripts.sample import ddpm_sampler
from TabDDPM.data.dataset import * 
from TabDDPM.data.data_utils import * 

# parser = argparse.ArgumentParser()
# parser.add_argument('ds_name', type=str)
# parser.add_argument('device', type = str, default = 'cuda:0')
# parser.add_argument('prefix', type=str)
# parser.add_argument('dp_epsilon', type=float, default=None)
# parser.add_argument('rare_threshold', type=float, default=0.005)
# parser.add_argument('--eval_seeds', action='store_true',  default=False)

# parser.add_argument('--syn_type', type=str, default = 'merf')
# parser.add_argument('--filter', action='store_true',  default=False)
# parser.add_argument('--aug', action='store_true',  default=False)

# simple_train = True: don't use pretrain model
# simple_train = False, skip_pretrain = True: use saved pretrained model
# simple_train = False, skip_pretrain = False: pretrain a model and use it (default case)
# parser.add_argument('--simple_train', action='store_true', default=False)
# parser.add_argument('--skip_pretrain',action='store_true', default=False)
# parser.add_argument('--skip_finetune',action='store_true', default=False)



def ddpm_main(args, df, domain, rho, parent_dir, **kwargs):

    if args.epsilon > 0:
        epsilon = rho*args.epsilon
        delta = rho*args.delta 
    else:
        epsilon = None 
        delta = None
    
    print(f'training privacy budget: ({epsilon},{delta})')

    base_config_path = f'TabDDPM/exp/{args.dataset}/config.toml'
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
    


    # if args.eval_seeds:
    #     print('-'*100)
    #     eval_seeds(
    #         base_config,
    #         dataset = dataset,
    #         n_seeds = 1,
    #         sampling_method = 'ddpm',
    #         n_datasets= 5,
    #         device = device
    #     )



#########################################################################

# unused scripts 

'''
if (not args.skip_pretrain) & (not args.simple_train):
    """
    decide the divide of privacy budget for each step
    imputation = 20%
    filter = 20%
    train = 1 - 20% - 20%
    if one step is not needed, the budget of that part is 0%
    """
    train_divide = 0.8
    if args.syn_type == 'aim':
        dataset.syn_pretrain_data(
            epsilon = (1-train_divide) * dp_epsilon, 
            delta = (1-train_divide) * 1e-5,
            sample_num = base_config['preprocess']['pretrain_sample_num']
        )
    elif args.syn_type == 'merf':
        dataset.syn_pretrain_data_merf(
            epsilon = (1-train_divide) * dp_epsilon, 
            delta = (1-train_divide) * 1e-5,
            device = device,
            sample_num = base_config['preprocess']['pretrain_sample_num']
        )

    if args.filter: 
        if_imputation = decide_imputation(dataset)
        if_preprocess = int(args.filter or args.augment) 
        train_divide = 1.0 - if_imputation * 0.2 - if_preprocess * 0.2
        if train_divide == 1.0:
            preprocess_divide_rho = 0
            imputation_divide_rho = 0
        else: 
            preprocess_divide_rho = if_preprocess / (if_preprocess + if_imputation)
            imputation_divide_rho = if_imputation / (if_preprocess + if_imputation)
            rho = calculate_rho((1 - train_divide) * dp_epsilon, (1 - train_divide) * 1e-5)

        base_config['preprocess']['contain_missing'] = bool(if_imputation) 

        # data imputation
        if imputation_divide_rho > 0:
            dataset.pretrain_data_imputation(
                rho = imputation_divide_rho * rho,
                margin_all = False
            )
        else:
            print('No data imputation needed')

        # filter & augmentation
        if preprocess_divide_rho > 0:
            initial_pretrain_size = len(dataset.y['pretrain'])
            data_processer = BlockRandomFourierFeatureProcesser(
                dataset = deepcopy(dataset),
                feature_dim = base_config['preprocess']['feature_dim'],
                rho = preprocess_divide_rho * rho,
                block_size = base_config['preprocess']['block_size'] if args.block_size is None else args.block_size
            )
            data_processer.initialize_fourier_feature()
            
            if args.filter:
                data_processer.RFF_greedy_filter_process(
                    str(parent_path / f'{prefix}_best/'), 
                    base_config['preprocess']['filter_threshold_param']
                )
            if args.aug:
                data_processer.RFF_genetic_aug_process()

            dataset.update_pretrain_data(data_processer.filter_idx, data_processer.aug_data_all)
            base_config['preprocess']['feature_dim'] = data_processer.feature_dim
            if args.block_size is not None:
                base_config['preprocess']['block_size'] = args.block_size
        else: 
            print('No pretrain data preprocess')
 

    pretrain_config = deepcopy(base_config)
    pretrain_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

    pretrain(
            **pretrain_config['pretrain']['main'],
            **pretrain_config['diffusion_params'],
            parent_dir=pretrain_config['parent_dir'],
            dataset = dataset,
            model_type=pretrain_config['model_type'],
            model_params=pretrain_config['model_params'],
            T_dict=pretrain_config['train']['T'],
            num_numerical_features=pretrain_config['num_numerical_features'],
            device=device
        )
'''