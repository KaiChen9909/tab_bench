import numpy as np
import os 
import tempfile
import shutil
import random
# from TabDDPM.scripts.sample import sample
from copy import deepcopy
from pathlib import Path
from evaluator.data.dataset import read_pure_data
from evaluator.data.data_utils import *
# from DP_MERF.sample import merf_heterogeneous_sample

def query_succeed(x: np.array, query_attr, query, query_type):
    query_res = np.full(len(x), True)
    for i in range(len(query)):
        if query_type[i] == 'num':
            query_res = query_res & (x[:, query_attr[i]].astype(float) >= query[i][0]) & (x[:, query_attr[i]].astype(float) <= query[i][1])
        elif query_type[i] == 'cat':
            query_res = query_res & (x[:, query_attr[i]] == query[i][0])
    return sum(query_res)

def make_query(
    synthetic_data_path,
    data_path,
    task_type,
    query_times,
    attr_num,
    seeds = 0
):
    random.seed(seeds)
    print("-" * 100)
    print('Starting query error evaluation')

    X_num_real, X_cat_real, y_real = read_pure_data(data_path, split = 'test')
    y_real = y_real.astype(int)
    if X_num_real is None: 
        X_cat_real = X_cat_real.astype(str)
        real_data = np.concatenate((X_cat_real, y_real.reshape(-1,1)), axis=1)
    elif X_cat_real is None: 
        X_num_real = X_num_real.astype(float)
        real_data = np.concatenate((X_num_real, y_real.reshape(-1,1)), axis=1)
    else:
        X_num_real = X_num_real.astype(float)
        X_cat_real = X_cat_real.astype(str)
        real_data = np.concatenate((X_num_real, X_cat_real, y_real.reshape(-1,1)), axis=1)

    X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, split = 'train')
    y_fake = y_fake.astype(int)
    if X_num_fake is None:
        X_cat_fake = X_cat_fake.astype(str)
        fake_data = np.concatenate((X_cat_fake, y_fake.reshape(-1,1)), axis=1)
    elif X_cat_fake is None: 
        X_num_fake = X_num_fake.astype(float)
        fake_data = np.concatenate((X_num_fake, y_fake.reshape(-1,1)), axis=1)
    else:
        X_num_fake = X_num_fake.astype(float)
        X_cat_fake = X_cat_fake.astype(str)
        fake_data = np.concatenate((X_num_fake, X_cat_fake, y_fake.reshape(-1,1)), axis=1)

    # obtain the domain of variables
    num_attr = 0
    cat_attr = 0
    num_range = None
    cat_range = None
    if X_num_real is not None: 
        num_attr = X_num_real.shape[1]
        num_range = get_numerical_range(X_num_real)
    if X_cat_real is not None: 
        cat_attr = X_cat_real.shape[1]
        cat_range = get_category_range(X_cat_real)
    y_range = get_numerical_range(y_real.reshape(-1,1)) if task_type == 'regression' else get_category_range(y_real.reshape(-1,1))

    error = []
    for i in range(query_times):
        # in each query time, choose attr respectively
        query_attr = np.random.choice(np.arange(0, num_attr + cat_attr + 1), size = attr_num, replace = False)
        # real_query_data = real_data[:, query_attr]
        # fake_query_data = fake_data[:, query_attr]
        query = []
        query_type = []
        for x in query_attr:
            if x < num_attr:
                query.append(sorted([random.uniform(num_range[x][0], num_range[x][1]), random.uniform(num_range[x][0], num_range[x][1])]))
                query_type.append('num')
            elif x >= num_attr and x < (num_attr + cat_attr): 
                query.append(np.random.choice(cat_range[x - num_attr], 1, replace=False))
                query_type.append('cat')
            else:
                if task_type == 'regression':
                    query.append(sorted([random.uniform(y_range[0][0], y_range[0][1]), random.uniform(y_range[0][0], y_range[0][1])]))
                    query_type.append('num')
                else:
                    query.append(np.random.choice(y_range[0], 1, replace=False))
                    query_type.append('cat')
            
            error.append(abs(
                query_succeed(real_data, query_attr, query, query_type)/len(real_data) - 
                query_succeed(fake_data, query_attr, query, query_type)/len(fake_data)
            ))
    
    print('query error:', np.mean(error))
    return np.mean(error)


# def query_error(
#         raw_config,
#         dataset,
#         device,
#         attr_num = 3,
#         n_datasets = 5,
#         query_times = 1000,
#         sampling_method = 'ddpm',
#         dp = True,
#         merf_dict = None,
#         merf_rare_dict = None,
#         privsyn_method = None,
#         privsyn_preprocess = None
# ):  
#     total_error = []
#     temp_config = deepcopy(raw_config)
#     raw_parent_dir = Path(raw_config['parent_dir'])
#     info = load_json(os.path.join(raw_config['real_data_path'], 'info.json'))
#     task_type = info['task_type']
#     ds = raw_config['real_data_path'].split('/')[-2]
    
#     with tempfile.TemporaryDirectory() as dir_:
#         temp_config['parent_dir'] = Path(dir_)
#         if sampling_method == "merf":
#             shutil.copy2(raw_parent_dir / "merf_model.pt", temp_config["parent_dir"])
#         elif sampling_method == 'privsyn':
#             shutil.copy2(f'privsyn/temp_data/processed_data/{ds}_mapping', temp_config["parent_dir"])
#         else:
#             shutil.copy2(raw_parent_dir / "model.pt", temp_config["parent_dir"])

#         for sample_seed in range(n_datasets):
#             temp_config['sample']['seed'] = sample_seed 
#             if sampling_method == 'merf':
#                 merf_heterogeneous_sample(
#                         **merf_dict,
#                         parent_dir = temp_config['parent_dir'],
#                         device = device,
#                         cat_rare_dict = merf_rare_dict
#                     )
            
#             elif sampling_method == 'privsyn':
#                     privsyn_method.synthesize_records()
#                     privsyn_method.postprocessing(dir_) # save raw synthesized data to temp dir

#                     privsyn_preprocess.reverse_mapping_from_files(temp_config['synthesized_filename'], temp_config['mapping_filename'], str(dir_)) 
#                     privsyn_preprocess.save_data_npy(dir_) # save reverse preprocessed synthesized data into temp dir

#             else:
#                 sample(
#                         num_samples=temp_config['sample']['num_samples'],
#                         batch_size=temp_config['sample']['batch_size'],
#                         disbalance=temp_config['sample'].get('disbalance', None),
#                         **temp_config['diffusion_params'],
#                         parent_dir=temp_config['parent_dir'],
#                         dataset = dataset,
#                         data_path = temp_config['real_data_path'],
#                         model_path=os.path.join(temp_config['parent_dir'], f'model.pt'),
#                         model_type=temp_config['model_type'],
#                         model_params=temp_config['model_params'],
#                         T_dict=temp_config['train']['T'],
#                         num_numerical_features=temp_config['num_numerical_features'],
#                         device=device,
#                         seed=temp_config['sample'].get('seed', 0),
#                         dp = dp
#                     )

#             synthetic_data_path = temp_config['parent_dir']
#             data_path = temp_config['real_data_path']

#             total_error.append(make_query(
#                 synthetic_data_path,
#                 data_path,
#                 task_type,
#                 query_times,
#                 attr_num
#             ))
#             print(f'Query evaluation round {sample_seed+1} finished')
#         shutil.rmtree(dir_)
    
#     return total_error


# def eval_query(
#         raw_config,
#         device = 'cuda:0',
#         dataset = None,
#         attr_num = 3,
#         n_datasets = 5,
#         query_times = 1000,
#         sampling_method = 'ddpm',
#         merf_dict = None,
#         merf_rare_dict = None,
#         privsyn_method = None,
#         privsyn_preprocess = None
# ):
#     parent_dir = raw_config['parent_dir']
#     dp = (sampling_method != 'ddpm')

#     total_error = query_error(
#             raw_config = raw_config,
#             dataset = dataset,
#             device = device,
#             attr_num = attr_num,
#             n_datasets = n_datasets,
#             query_times = query_times,
#             sampling_method = sampling_method,
#             dp = dp,
#             merf_dict = merf_dict,
#             merf_rare_dict = merf_rare_dict,
#             privsyn_method = privsyn_method,
#             privsyn_preprocess = privsyn_preprocess
#     )
    
#     eval_dict = {
#         'n_datasets' : n_datasets,
#         'eval_times' : query_times,
#         'error_mean' : np.mean(total_error)
#     }
#     print('='*100)
#     print(eval_dict)
#     print('='*100)

#     dump_json(eval_dict, os.path.join(parent_dir, 'eval_query.json'))

        
