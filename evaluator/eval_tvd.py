import numpy as np
import os 
import tempfile
import shutil
import random
import itertools
# from TabDDPM.scripts.sample import sample
from copy import deepcopy
from collections import Counter
from pathlib import Path
from evaluator.data.dataset import read_pure_data
from evaluator.data.data_utils import dump_json 


def num_divide(X_num_real, X_num_fake):
    for i in range(X_num_fake.shape[1]):
        max_value = max([max(X_num_real[:, i]), max(X_num_fake[:, i])])
        min_value = min([min(X_num_real[:, i]), min(X_num_fake[:, i])])

        X_num_real[:, i] = np.round(99 * (X_num_real[:, i] - min_value)/(max_value - min_value))
        X_num_fake[:, i] = np.round(99 * (X_num_fake[:, i] - min_value)/(max_value - min_value)) 

    return X_num_real, X_num_fake


def marginal_TVD(real_data, syn_data, col, dimension = 1):
    """
    real_probs: list of marginal probabilities for real data
    syn_probs: list of marginal probabilities for syn data
    calulate the average absolute difference between real_probs and syn_probs
    """
    if dimension == 1:
        count_real_dict = Counter(real_data[:,col])
        count_syn_dict = Counter(syn_data[:,col])
    elif dimension == 2:
        count_real_dict = Counter(zip(real_data[:,col[0]], real_data[:,col[1]]))
        count_syn_dict = Counter(zip(syn_data[:,col[0]], syn_data[:,col[1]]))
    elif dimension == 3:
        count_real_dict = Counter(zip(real_data[:,col[0]], real_data[:,col[1]], real_data[:,col[2]]))
        count_syn_dict = Counter(zip(syn_data[:,col[0]], syn_data[:,col[1]], syn_data[:,col[2]]))
    else:
        raise 'Unsupported margin dimension'
    all_value_set = count_real_dict.keys() | count_syn_dict.keys()

    real_probs = []
    syn_probs = []
    for x in all_value_set:
        real_probs.append(count_real_dict[x])
        syn_probs.append(count_syn_dict[x])

    sum_real_probs = sum(real_probs)
    real_probs = [x/sum_real_probs for x in real_probs]

    sum_syn_probs = sum(syn_probs)
    syn_probs = [x/sum_syn_probs for x in syn_probs]

    try:
        assert sum(real_probs) >= 1 - 1e-2
        assert sum(real_probs) <= 1 + 1e-2
        assert sum(syn_probs) >= 1 - 1e-2
        assert sum(syn_probs) <= 1 + 1e-2
    except:
        print("error in marginal_query for cols: ", col)
        print("real_probs: ", sum(real_probs))
        print("syn_probs: ", sum(syn_probs))
        raise ValueError("sum of probs should be 1")

    abs_diff = np.abs(np.array(real_probs) - np.array(syn_probs))/2
    return sum(abs_diff)


def tvd_main(
    X_num_real, X_cat_real, y_real,
    X_num_fake, X_cat_fake, y_fake,
    dim = None
):
    if X_num_real is not None:
        X_num_real, X_num_fake = num_divide(X_num_real, X_num_fake)

    if X_num_real is None: 
        real_data = np.concatenate((X_cat_real, y_real.reshape(-1,1)), axis=1).astype(str)
    elif X_cat_real is None: 
        real_data = np.concatenate((X_num_real, y_real.reshape(-1,1)), axis=1).astype(str)
    else:
        real_data = np.concatenate((X_num_real, X_cat_real, y_real.reshape(-1,1)), axis=1).astype(str)

    if X_num_fake is None:
        fake_data = np.concatenate((X_cat_fake, y_fake.reshape(-1,1)), axis=1).astype(str)
    elif X_cat_fake is None: 
        fake_data = np.concatenate((X_num_fake, y_fake.reshape(-1,1)), axis=1).astype(str)
    else:
        fake_data = np.concatenate((X_num_fake, X_cat_fake, y_fake.reshape(-1,1)), axis=1).astype(str)

    data_shape = real_data.shape[1]

    l1_list = {'1way margin': -1, '2way margin': -1}
    if dim == 2:
        margin_error_comb = []
        attr_num = 2
        
        combinations = tuple(itertools.combinations(np.arange(data_shape), attr_num))
        for combination in combinations:
            margin_error_comb.append(
                marginal_TVD(real_data, fake_data, combination, attr_num)
            )

        l1_list[f'{attr_num}way margin'] = np.mean(margin_error_comb)
        print(f'finish {attr_num}-way marigin TVD evaluation, error is', np.mean(margin_error_comb))
    else:
        for attr_num in range(1,3):
            margin_error_comb = []

            if attr_num == 1:
                combinations = tuple(np.arange(real_data.shape[1]))
            else:
                combinations = tuple(itertools.combinations(np.arange(data_shape), attr_num))
            
            for combination in combinations:
                margin_error_comb.append(
                    marginal_TVD(real_data, fake_data, combination, attr_num)
                )

            l1_list[f'{attr_num}way margin'] = np.mean(margin_error_comb)
            print(f'finish {attr_num}-way marigin TVD evaluation, error is', np.mean(margin_error_comb))
    
    return l1_list


def tvd_divide(
    X_num_real, X_cat_real, y_real,
    X_num_fake, X_cat_fake, y_fake,
    dim = None, part = 'all'
):
    num_id = 0
    cat_id = 0
    if X_num_real is not None:
        X_num_real, X_num_fake = num_divide(X_num_real, X_num_fake)

    if X_num_real is None: 
        cat_id = X_cat_real.shape[1]
        real_data = np.concatenate((X_cat_real, y_real.reshape(-1,1)), axis=1).astype(str)
    elif X_cat_real is None: 
        num_id = X_num_real.shape[1]
        real_data = np.concatenate((X_num_real, y_real.reshape(-1,1)), axis=1).astype(str)
    else:
        num_id = X_num_real.shape[1]
        cat_id = X_cat_real.shape[1]
        real_data = np.concatenate((X_num_real, X_cat_real, y_real.reshape(-1,1)), axis=1).astype(str)

    if X_num_fake is None:
        fake_data = np.concatenate((X_cat_fake, y_fake.reshape(-1,1)), axis=1).astype(str)
    elif X_cat_fake is None: 
        fake_data = np.concatenate((X_num_fake, y_fake.reshape(-1,1)), axis=1).astype(str)
    else:
        fake_data = np.concatenate((X_num_fake, X_cat_fake, y_fake.reshape(-1,1)), axis=1).astype(str)

    data_shape = real_data.shape[1]

    if dim == 2:
        l1_list = {'2way margin':[]}
        margin_error_comb = []
        attr_num = 2
        
        if part == 'all':
            combinations = tuple(itertools.combinations(np.arange(data_shape), attr_num))
        else:
            combinations = ()
            if 'num-cat' in part:
                combinations += tuple(itertools.product(range(0, num_id), range(num_id, num_id + cat_id)))
            if 'num-num' in part:
                combinations += tuple(itertools.combinations(np.arange(num_id), attr_num))
            if 'cat-cat' in part :
                combinations += tuple(itertools.combinations(np.arange(num_id, num_id + cat_id), attr_num))
            if 'num-y' in part:
                combinations += tuple(itertools.product(range(0, num_id), range(num_id + cat_id, num_id + cat_id + 1)))
            if 'cat-y' in part:
                combinations += tuple(itertools.product(np.arange(num_id, num_id + cat_id), range(num_id + cat_id, num_id + cat_id + 1)))
            if 'num-caty' in part:
                combinations += tuple(itertools.product(range(0, num_id), range(num_id, num_id + cat_id + 1)))
            if 'num-num' in part:
                combinations += tuple(itertools.combinations(np.arange(num_id), attr_num))
            if 'caty-caty' in part :
                combinations += tuple(itertools.combinations(np.arange(num_id, num_id + cat_id + 1), attr_num))
            assert len(combinations) > 0

        for combination in combinations:
            margin_error_comb.append(
                marginal_TVD(real_data, fake_data, combination, attr_num)
            )

        l1_list[f'{attr_num}way margin'].append(np.mean(margin_error_comb))
        print(f'finish {attr_num}-way marigin TVD evaluation, error is', np.mean(margin_error_comb))
    else:
        l1_list = {'1way margin':[], '2way margin':[]}
        for attr_num in range(1,3):
            margin_error_comb = []

            if attr_num == 1:
                combinations = tuple(np.arange(real_data.shape[1]))
            else:
                combinations = tuple(itertools.combinations(np.arange(data_shape), attr_num))
            
            for combination in combinations:
                margin_error_comb.append(
                    marginal_TVD(real_data, fake_data, combination, attr_num)
                )

            l1_list[f'{attr_num}way margin'].append(np.mean(margin_error_comb))
            print(f'finish {attr_num}-way marigin TVD evaluation, error is', np.mean(margin_error_comb))
    
    return l1_list



def make_tvd(
    synthetic_data_path,
    data_path
):  
    print('-' * 100)
    print('Starting TVD evaluation')
    X_num_real, X_cat_real, y_real = read_pure_data(data_path, split = 'test')
    X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, split = 'train') 
    
    return tvd_main(
        X_num_real, X_cat_real, y_real,
        X_num_fake, X_cat_fake, y_fake
    )