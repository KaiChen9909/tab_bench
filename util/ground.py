# This is a auxliary file for obtaining 
import sys 
import argparse
import os
from pathlib import Path
from copy import deepcopy
from TabDDPM.data.data_utils import * 
from TabDDPM.data.metrics import * 
from evaluator.eval_catboost import train_catboost
from evaluator.eval_mlp import train_mlp
from evaluator.eval_transformer import train_transformer
from evaluator.eval_simple import train_simple 
from evaluator.eval_tvd import make_tvd
from evaluator.eval_query import make_query

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
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
            parent_dir = None,
            data_path=temp_config['real_data_path'],
            eval_type='real',
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        )
    
    elif model_type == "mlp":
        T_dict["normalization"] = "quantile"
        T_dict["cat_encoding"] = "one-hot"
        metric_report = train_mlp(
            # parent_dir=temp_config['parent_dir'],
            parent_dir = None,
            data_path=temp_config['real_data_path'],
            eval_type='real',
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        )

    elif model_type == "transformer":
        T_dict["normalization"] = "quantile"
        T_dict["cat_encoding"] = "one-hot"
        metric_report = train_transformer(
            # parent_dir=temp_config['parent_dir'],
            parent_dir = None,
            data_path=temp_config['real_data_path'],
            eval_type='real',
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        )
    
    else:
        T_dict["normalization"] = "minmax"
        T_dict["cat_encoding"] = None
        metric_report = train_simple(
            # parent_dir=temp_config['parent_dir'],
            parent_dir = None,
            data_path=temp_config['real_data_path'],
            eval_type='real',
            model_name=model_type,
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        ) 

    return metric_report


def main(args):
    n_seeds=5

    parent_dir = Path(f'exp/{args.ds_name}/Ground')
    os.makedirs(parent_dir, exist_ok=True)
    data_path = Path(f"data/{args.ds_name}")
    info = load_json(os.path.join(data_path, 'info.json'))
    task_type = info['task_type'] 

    temp_config = {
        'parent_dir': data_path,
        'real_data_path': data_path,
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
        'catboost': os.path.exists(f'eval_models/catboost/{args.ds_name}_cv.json'), 
        'mlp': os.path.exists(f'eval_models/mlp/{args.ds_name}_cv.json'), 
        'rf': os.path.exists(f'eval_models/rf/{args.ds_name}_cv.json'),
        'xgb': os.path.exists(f'eval_models/xgb/{args.ds_name}_cv.json')
    } 

    for seed in range(n_seeds):
        for model_type in ['catboost', 'mlp', 'rf', 'xgb']: 
            if not eval_support[model_type]:
                continue
            else:
                metric_report = prepare_report(model_type, temp_config, seed)
                metrics_seeds_report[model_type].add_report(metric_report)

        query_report.append(make_query(
            data_path,
            data_path,
            task_type,
            query_times = 1000,
            attr_num = 3,
            seeds = seed
        ))

        tvd_error = make_tvd(data_path, data_path) 
        if not tvd_report:
            for k,v in tvd_error.items():
                tvd_report[k] = [v] 
        else:
            for k,v in tvd_error.items():
                tvd_report[k].append(v) 
    
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
    

    # summarize query result
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

if __name__ == '__main__':
    main(args)