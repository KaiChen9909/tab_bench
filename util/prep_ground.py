############################################################
#
# This file is used to calculate the error of preprocessing
#
############################################################
import os 
import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import pandas as pd
import json 
import os 
import argparse
import itertools
import tempfile
from copy import deepcopy
from preprocess_common.preprocess import *
from rho_cdp import cdp_rho
from TabDDPM.data.data_utils import * 
from TabDDPM.data.metrics import * 
from evaluator.eval_catboost import train_catboost
from evaluator.eval_mlp import train_mlp
from evaluator.eval_transformer import train_transformer
from evaluator.eval_simple import train_simple 
from evaluator.eval_tvd import make_tvd
from evaluator.eval_query import make_query

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('device', type=str)

args = parser.parse_args()

def prepare_report(model_type, temp_config, seed):
    T_dict = {
        'seed': 0,
        'normalization': "quantile",
        'num_nan_policy': None,
        'cat_nan_policy': None,
        'cat_min_frequency': None,
        'cat_encoding': "one-hot",
        'y_policy': "default"
    }
    
    if model_type == "catboost":
        T_dict["normalization"] = None
        T_dict["cat_encoding"] = None
        metric_report = train_catboost(
            # parent_dir=temp_config['parent_dir'],
            parent_dir = temp_config['parent_dir'],
            data_path=temp_config['real_data_path'],
            eval_type='synthetic',
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        )
    
    elif model_type == "mlp":
        T_dict["normalization"] = "quantile"
        T_dict["cat_encoding"] = "one-hot"
        metric_report = train_mlp(
            # parent_dir=temp_config['parent_dir'],
            parent_dir = temp_config['parent_dir'],
            data_path=temp_config['real_data_path'],
            eval_type='synthetic',
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        )

    elif model_type == "transformer":
        T_dict["normalization"] = "quantile"
        T_dict["cat_encoding"] = "one-hot"
        metric_report = train_transformer(
            # parent_dir=temp_config['parent_dir'],
            parent_dir = temp_config['parent_dir'],
            data_path=temp_config['real_data_path'],
            eval_type='synthetic',
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        )
    
    else:
        T_dict["normalization"] = "minmax"
        T_dict["cat_encoding"] = None
        metric_report = train_simple(
            # parent_dir=temp_config['parent_dir'],
            parent_dir = temp_config['parent_dir'],
            data_path=temp_config['real_data_path'],
            eval_type='synthetic',
            model_name=model_type,
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        ) 

    return metric_report




def preprocess_error(dataset):
    path = f'data/{dataset}/'
    eps = 1.0 
    delta = 1e-5
    rho = cdp_rho(eps, delta)

    X_num = None
    X_cat = None
    X_num_p = None 
    X_num_pp = None 
    X_cat_p = None 
    X_cat_pp = None 
    
    if os.path.exists(os.path.join(path, 'X_num_train.npy')):
        X_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, 'X_cat_train.npy')):
        X_cat = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)
    y = np.load(os.path.join(path, 'y_train.npy'), allow_pickle=True)
    num_divide, cat_divide = calculate_rho_allocate(X_num, X_cat, 'uniform_kbins')

    if X_num is not None:
        num_encoder = discretizer('uniform_kbins', num_divide * 0.1 * rho)
        X_num_p = num_encoder.fit_transform(X_num) 
        X_num_pp = num_encoder.inverse_transform(X_num_p)

    if X_cat is not None:
        rare_threshold = 0.002
        cat_encoder = rare_merger(cat_divide * 0.1 * rho, rare_threshold=rare_threshold, output_type='ordinal')
        X_cat_p = cat_encoder.fit_transform(X_cat)
        X_cat_pp = cat_encoder.inverse_transform(X_cat_p)

    if X_num is not None and X_cat is not None:
        data = np.concatenate((X_num, X_cat, y.reshape(-1,1)), axis=1)
        data_n_p = np.concatenate((X_num_p, X_cat, y.reshape(-1,1)), axis=1)
        data_c_p = np.concatenate((X_num, X_cat_p, y.reshape(-1,1)), axis=1)
        data_p = np.concatenate((X_num_p, X_cat_p, y.reshape(-1,1)), axis=1)
    elif X_num is None:
        data = np.concatenate((X_cat, y.reshape(-1,1)), axis=1)
        data_n_p = np.concatenate((X_cat, y.reshape(-1,1)), axis=1)
        data_c_p = np.concatenate((X_cat_p, y.reshape(-1,1)), axis=1)
        data_p = np.concatenate((X_cat_p, y.reshape(-1,1)), axis=1) 
    elif X_cat is None:
        data = np.concatenate((X_num, y.reshape(-1,1)), axis=1)
        data_n_p = np.concatenate((X_num_p, y.reshape(-1,1)), axis=1)
        data_c_p = np.concatenate((X_num, y.reshape(-1,1)), axis=1)
        data_p = np.concatenate((X_num_p, y.reshape(-1,1)), axis=1) 

    marginals = list(itertools.combinations(np.arange(data.shape[1]), 2))

    marginal_size = []
    marginal_size_n_p = []
    marginal_size_c_p = []
    marginal_size_p = []

    for a,b in marginals:
        marginal_size.append(len(set(data[:,a])) * len(set(data[:,b])))
        marginal_size_n_p.append(len(set(data_n_p[:,a])) * len(set(data_n_p[:,b])))
        marginal_size_c_p.append(len(set(data_c_p[:,a])) * len(set(data_c_p[:,b])))
        marginal_size_p.append(len(set(data_p[:,a])) * len(set(data_p[:,b])))
    
    os.makedirs(f'exp/{dataset}/Ground', exist_ok=True)
    os.makedirs(f'exp/{dataset}/num_Ground', exist_ok=True)
    os.makedirs(f'exp/{dataset}/cat_Ground', exist_ok=True)
    os.makedirs(f'exp/{dataset}/all_Ground', exist_ok=True)

    with open(f'exp/{dataset}/Ground/marginal.json', 'w') as file:
        json.dump({'size': marginal_size}, file, indent=4)
    with open(f'exp/{dataset}/num_Ground/marginal.json', 'w') as file:
        json.dump({'size': marginal_size_n_p}, file, indent=4)
    with open(f'exp/{dataset}/cat_Ground/marginal.json', 'w') as file:
        json.dump({'size': marginal_size_c_p}, file, indent=4)
    with open(f'exp/{dataset}/all_Ground/marginal.json', 'w') as file:
        json.dump({'size': marginal_size_p}, file, indent=4)
    
    for fix in ['num', 'cat', 'all']:
        parent_dir = Path(f'exp/{dataset}/{fix}_Ground')
        data_path = Path(f'data/{dataset}')
        info = load_json(os.path.join(data_path, 'info.json'))
        task_type = info['task_type'] 

        temp_config = {
            'parent_dir': str(parent_dir),
            'real_data_path': str(data_path),
            'model_params':{'num_classes': info['n_classes']},
            'sample': {'seed': 0, 'sample_num': info['train_size']}
        }

        metrics_seeds_report = {
            'catboost': SeedsMetricsReport(), 
            'mlp': SeedsMetricsReport(), 
            'rf': SeedsMetricsReport(),
            'xgb': SeedsMetricsReport()
        }
        query_report = []
        tvd_report = {}

        eval_support = {
            'catboost': os.path.exists(f'eval_models/catboost/{dataset}_cv.json'), 
            'mlp': os.path.exists(f'eval_models/mlp/{dataset}_cv.json'), 
            'rf': os.path.exists(f'eval_models/rf/{dataset}_cv.json'),
            'xgb': os.path.exists(f'eval_models/xgb/{dataset}_cv.json')
        } 

        
        with tempfile.TemporaryDirectory() as dir_:
            n_seeds=5
            dir_ = Path(dir_)
            temp_config["parent_dir"] = str(dir_)

            np.save(os.path.join(dir_, 'y_train.npy'), y)
            if fix == 'num':
                if X_num is not None:
                    np.save(os.path.join(dir_, 'X_num_train.npy'), X_num_pp)
                if X_cat is not None:
                    np.save(os.path.join(dir_, 'X_cat_train.npy'), X_cat)
            elif fix == 'cat':
                if X_num is not None:
                    np.save(os.path.join(dir_, 'X_num_train.npy'), X_num)
                if X_cat is not None:
                    np.save(os.path.join(dir_, 'X_cat_train.npy'), X_cat_pp)
            elif fix == 'all':
                if X_num is not None:
                    np.save(os.path.join(dir_, 'X_num_train.npy'), X_num_pp)
                if X_cat is not None:
                    np.save(os.path.join(dir_, 'X_cat_train.npy'), X_cat_pp)

            for seed in range(n_seeds): 
                for model_type in ['catboost', 'mlp', 'rf', 'xgb']: 
                    if not eval_support[model_type]:
                        continue
                    else:
                        metric_report = prepare_report(model_type, temp_config, seed)
                        metrics_seeds_report[model_type].add_report(metric_report)

                query_report.append(make_query(
                    temp_config["parent_dir"],
                    data_path,
                    task_type,
                    query_times = 1000,
                    attr_num = 3,
                    seeds = seed
                ))

                tvd_error = make_tvd(temp_config["parent_dir"], data_path) 
                if not tvd_report:
                    for k,v in tvd_error.items():
                        tvd_report[k] = [v] 
                else:
                    for k,v in tvd_error.items():
                        tvd_report[k].append(v)

        # parent_dir = Path(parent_dir)
        for model_type in ['catboost', 'mlp', 'rf', 'xgb']:
            if eval_support[model_type]:
                metrics_seeds_report[model_type].get_mean_std()
                res = metrics_seeds_report[model_type].print_result()

                if os.path.exists(parent_dir/ f"eval_{model_type}.json"):
                    eval_dict = load_json(parent_dir / f"eval_{model_type}.json")
                    eval_dict = eval_dict | {'synthetic': res}
                else:
                    eval_dict = {'synthetic': res}
                
                dump_json(eval_dict, parent_dir / f"eval_{model_type}.json")
            else:
                print(f'{model_type} evaluation is not supported for this dataset')   

        query_report_final = {
            'n_datasets' : 1,
            'eval_times' : 1000,
            'error_mean' : np.mean(query_report)
        }
        print('query error evaluation:')
        print(query_report_final)
        if os.path.exists(parent_dir/ f"eval_query.json"):
            eval_dict = load_json(parent_dir / f"eval_query.json")
            eval_dict = eval_dict | {'synthetic': query_report_final}
        else: 
            eval_dict = {'synthetic': query_report_final}
        
        dump_json(eval_dict, os.path.join(parent_dir, 'eval_query.json'))
        

        # summarize l1 result
        tvd_report_final = {}
        for k,v in tvd_report.items():
            tvd_report_final[k] = {}
            tvd_report_final[k]['mean'] = np.mean(tvd_report[k])
            tvd_report_final[k]['std'] = np.std(tvd_report[k]) 

        print('='*100)
        print('tvd error evaluation:')
        print(tvd_report_final)
        if os.path.exists(parent_dir/ f"eval_tvd.json"):
            eval_dict = load_json(parent_dir / f"eval_tvd.json")
            eval_dict = eval_dict | {'synthetic': tvd_report_final}
        else: 
            eval_dict = {'synthetic': tvd_report_final}

        dump_json(eval_dict, os.path.join(parent_dir, 'eval_tvd.json'))         

    return 0


def main(args):
    preprocess_error(args.dataset)


if __name__ == '__main__':
    main(args)