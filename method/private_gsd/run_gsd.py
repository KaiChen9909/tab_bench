import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import argparse
import os 
import json
import math
import sklearn.preprocessing

from method.private_gsd.utils.utils_data import Dataset, Domain
from method.private_gsd.stats import Marginals, ChainedStatistics
from method.private_gsd.models import GSD 
from method.private_gsd.utils.cdp2adp import cdp_rho
from jax.random import PRNGKey


# formatter = argparse.ArgumentDefaultsHelpFormatter
# parser = argparse.ArgumentParser(description="", formatter_class=formatter)
# parser.add_argument("dataset", help="dataset to use")
# parser.add_argument("device", help="device to use")
# parser.add_argument("epsilon", type=float, help="privacy parameter")
# parser.add_argument("--delta", type=float, help="privacy parameter", default=1e-5)
# parser.add_argument("--num_preprocess", type=str, default='privtree')
# parser.add_argument("--rare_threshold", type=float, default=0.005)

# args = parser.parse_args()

def prepare_domain(domain):
    domain_new = {}
    for k,v in domain.items():
        if k.split('_')[0] == 'num':
            domain_new[k] = 1
        else:
            domain_new[k] = v
    return domain_new

def prepare_df(df):
    num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
    
    num_idx = []
    for i in range(df.shape[1]):
        if df.columns[i].split('_')[0] == 'num':
            num_idx.append(i)
    
    df.iloc[:, num_idx] = num_encoder.fit_transform(df.iloc[:, num_idx])

    return df, num_encoder, num_idx

def gsd_main(args, df, domain, rho, **kwargs): 
    domain = Domain.fromdict(prepare_domain(domain))
    df, encoder, num_idx  = prepare_df(df)
    # domain = Domain.fromdict(domain)
    data = Dataset(df, domain)

    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = GSD(
        domain=data.domain,
        print_progress=True,
        stop_early=True,
        # num_generations=20000,
        population_size_muta=50,
        population_size_cross=50,
        # data_size = df.shape[0]
        num_encoder = encoder,
        num_idx = num_idx,
        args=args
    ) 

    key = PRNGKey(0)
    algo.zcdp_syn_init(key, stat_module, rho)

    return {'gsd_generator': algo}


'''
def determin_cat_type(method):
    if method in ['merf', 'ddpm']:
        return 'one_hot'
    else:
        return 'ordinal'

if __name__ == "__main__":
    data_path = f'data/{args.dataset}'

    total_rho = rdp_rho(args.epsilon, args.delta)
    data_preporcesser = data_preporcesser_common(cat_output_type=determin_cat_type('gsd'))
    df, domain, preprocesser_rho_divide  = data_preporcesser.load_data(data_path, total_rho, args.num_preprocess, args.rare_threshold) 

    gsd_dict = gsd_main(args, df, domain, total_rho)
    syn_df = gsd_dict['gsd_generator'].syn(2000, data_preporcesser)
    print(syn_df.shape)
    print(syn_df.head(5))

'''



# if not args.no_eval:
#     with open(f'data/{args.dataset}/info.json', 'r') as file:
#             data_info = json.load(file)
#     config = {'parent_dir': parent_dir,
#                 'real_data_path': f'data/{args.dataset}/',
#                 'model_params':{'num_classes': data_info['n_classes']},
#                 'sample': {'seed': 0, 'sample_num': data_info['train_size']}
#             }
    
#     with open(os.path.join(parent_dir, 'config.json'), 'w', encoding = 'utf-8') as file: 
#         json.dump(config, file)
#     eval_seeds(
#                 raw_config = config,
#                 n_seeds = 1,
#                 n_datasets = 5,
#                 device = args.device,
#                 sampling_method = 'gsd',
#                 gsd_algo = algo,
#                 gsd_syn_dict = gsd_syn_dict,
#                 data_preprocesser = data_loader
#             )