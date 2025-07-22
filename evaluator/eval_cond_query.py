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

def cond_query_succeed(x: np.array, query_attr, query, query_type):
    if query_type[0] == 'num':
        x_target = x[(x[:, query_attr[0]].astype(float) >= query[0][0]) & (x[:, query_attr[0]].astype(float) <= query[0][1])]
    else:
        x_target = x[x[:, query_attr[0]] == query[0][0]]

    query_res = np.full(len(x_target), True)
    for i in range(1, len(query)):
        if query_type[i] == 'num':
            query_res = query_res & (x_target[:, query_attr[i]].astype(float) >= query[i][0]) & (x_target[:, query_attr[i]].astype(float) <= query[i][1])
        elif query_type[i] == 'cat':
            query_res = query_res & (x_target[:, query_attr[i]] == query[i][0])
    
    return sum(query_res)/len(x_target) if len(x_target)>0 else 0

def make_cond_query(
    synthetic_data_path,
    data_path,
    task_type,
    query_times,
    seeds = 0
):
    random.seed(seeds)
    print("-" * 100)
    print('Starting conditional query error evaluation')

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
        query_attr = np.random.choice(np.arange(0, num_attr + cat_attr + 1), size = 2, replace = False)
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
                cond_query_succeed(real_data, query_attr, query, query_type) - 
                cond_query_succeed(fake_data, query_attr, query, query_type)
            ))
    
    print('query error:', np.mean(error))
    print('-'*100)
    return np.mean(error)