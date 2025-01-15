import tempfile
import os
import sys
target_path="./"
sys.path.append(target_path)

import shutil
import time
from pathlib import Path
from copy import deepcopy
from evaluator.data.data_utils import * 
from evaluator.data.metrics import * 
from evaluator.eval_catboost import train_catboost
from evaluator.eval_mlp import train_mlp
from evaluator.eval_transformer import train_transformer
from evaluator.eval_simple import train_simple 
from evaluator.eval_tvd import make_tvd
from evaluator.eval_query import make_query
from evaluator.eval_sample import eval_sampler


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
            parent_dir=temp_config['parent_dir'],
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
            parent_dir=temp_config['parent_dir'],
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
            parent_dir=temp_config['parent_dir'],
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
            parent_dir=temp_config['parent_dir'],
            data_path=temp_config['real_data_path'],
            eval_type='synthetic',
            model_name=model_type,
            T_dict=T_dict,
            seed=seed,
            change_val=False #False
        ) 
    
    return metric_report


def save_time_record(dict, path):
    with open(os.path.join(path, 'time.json'), 'w') as file:
        json.dump(dict, file, indent=4)


def eval_seeds(
    raw_config,
    n_seeds = 1,
    dataset = None,
    preprocesser = None,
    sampling_method="ddpm",
    n_datasets = 5,
    device='cuda:0',
    time_record=None,
    **kwargs, #these are some necessary params for baselines
    # merf_dict = None,
    # privsyn_method = None,
    # privsyn_postprocess = None,
    # aim_generator = None,
    # aim_dict = None,
    # llm_generator = None
):
    parent_dir = Path(raw_config["parent_dir"])

    info = load_json(os.path.join(raw_config['real_data_path'], 'info.json'))
    task_type = info['task_type']

    ds = raw_config['real_data_path'].split('/')[-2] # dataset name 

    # initialize metrics dict
    metrics_seeds_report = {
        'catboost': SeedsMetricsReport(), 
        'mlp': SeedsMetricsReport(), 
        'rf': SeedsMetricsReport(),
        'xgb': SeedsMetricsReport()
    }
    query_report = []
    tvd_report = {}

    # whether these eval model is supported
    eval_support = {
        'catboost': os.path.exists(f'eval_models/catboost/{ds}_cv.json'), 
        'mlp': os.path.exists(f'eval_models/mlp/{ds}_cv.json'), 
        'rf': os.path.exists(f'eval_models/rf/{ds}_cv.json'),
        'xgb': os.path.exists(f'eval_models/xgb/{ds}_cv.json')
    } 
    print(eval_support)


    temp_config = deepcopy(raw_config)
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        temp_config["parent_dir"] = str(dir_)
        shutil.copy2(parent_dir / "eval_config.json", temp_config["parent_dir"])
        time_all = 0.0

        for sample_seed in range(n_datasets):
            temp_config['sample']['seed'] = sample_seed
            if n_datasets > 1:
                start_time = time.time()
                eval_sampler(sampling_method, temp_config, device, preprocesser, **kwargs) # synthesize data
                end_time = time.time() 
                
                time_all += (start_time - end_time) # a typo, show be negative

            synthetic_data_path = temp_config['parent_dir']
            data_path = temp_config['real_data_path']
                        
            for seed in range(n_seeds):
                for model_type in ['catboost', 'mlp', 'rf', 'xgb']: 
                    if not eval_support[model_type]:
                        continue
                    else:
                        print(f'**Eval Iter: {sample_seed*n_seeds + (seed + 1)}/{n_seeds * n_datasets}**')
                        metric_report = prepare_report(model_type, temp_config, seed)
                        metrics_seeds_report[model_type].add_report(metric_report)

                query_report.append(make_query(
                    synthetic_data_path,
                    data_path,
                    task_type,
                    query_times = 1000,
                    attr_num = 3,
                    seeds = seed
                ))
                
                tvd_error = make_tvd(synthetic_data_path, data_path) 
                if not tvd_report:
                    for k,v in tvd_error.items():
                        tvd_report[k] = [v] 
                else:
                    for k,v in tvd_error.items():
                        tvd_report[k].append(v) 
                
    try:
        shutil.rmtree(dir_)
    except:
        print('No temp dir found')

    # summarize ml result
    time_record['synthesis time'] = time_all / n_datasets
    save_time_record(time_record, parent_dir)


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
        'n_datasets' : n_datasets,
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

