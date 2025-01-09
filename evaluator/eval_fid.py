import pandas as pd
import numpy as np
import ot
import os 
import tempfile
import shutil
import random
import warnings
import sklearn.preprocessing
from scipy.stats import wasserstein_distance
from collections import Counter
from TabDDPM.scripts.sample import sample
from copy import deepcopy
from pathlib import Path
from evaluator.data.dataset import read_pure_data
from evaluator.data.data_utils import * 
from DP_MERF.sample import merf_heterogeneous_sample



def cal_fidelity(real_data_num, real_data_cat, syn_data_num, syn_data_cat):
    # first, compute column-wise distance
    ret = {}

    if real_data_cat is not None: 
        print("computing cat_error")
        _cat_error = cat_error(real_data_cat, syn_data_cat)
        if _cat_error:
            ret["cat_error"] = _cat_error 
            print('cat fid:', _cat_error)
        else:
            print('no cat fid result')

    if real_data_num is not None: 
        print("computing num_error")
        _num_error = num_error(real_data_num, syn_data_num)
        if _num_error:
            ret["num_error"] = _num_error
            print('num fid:', _num_error)
        else:
            print('no num fid result')

    if real_data_num is not None: 
        print("computing num_num_error")
        _num_num_error = num_num_error(real_data_num, syn_data_num)
        if _num_num_error:
            ret["num_num_error"] = _num_num_error
            print('num-num fid:', _num_num_error)
        else:
            print('no num-num fid result')

    if (real_data_num is not None) & (real_data_cat is not None): 
        print("computing cat_num_error")
        _cat_num_error = cat_num_error(real_data_num, real_data_cat, syn_data_num, syn_data_cat)
        if _cat_num_error:
            ret["cat_num_error"] = _cat_num_error
            print("cat-num fid:", _cat_num_error)
        else:
            print('no cat-num fid result')

    if real_data_cat is not None: 
        print("computing cat_cat_error")
        _cat_cat_error = cat_cat_error(real_data_cat, syn_data_cat)
        if _cat_cat_error:
            ret["cat_cat_error"] = _cat_cat_error
            print("cat-cat fid:", _cat_cat_error)
        else:
            print('no cat-cat fid result')

    return ret


###########################################################################
#########################2-way Wasserstein distance########################
###########################################################################


def cat_num_error(real_data_num, real_data_cat, syn_data_num, syn_data_cat):
    """
    compute the cat-num error (2-way) for discrete and continuous columns between real and syn data
    """
    # for each combination of discrete and continuous columns, compute the 2D wasserstein distance
    wasserstein_error = []
    for i in range(real_data_cat.shape[1]):
        for j in range(real_data_num.shape[1]):
            real_cat = real_data_cat[:,i].reshape(-1, 1)
            syn_cat = syn_data_cat[:,i].reshape(-1, 1)
            real_num = real_data_num[:,j].reshape(-1, 1)
            syn_num = syn_data_num[:,j].reshape(-1, 1)

            # compute the categorical distance matrix 
            cat_dist_matrix = np.not_equal(real_cat[:, None], syn_cat).astype(int).squeeze()

            # normalize the numerical column to [0, 1]
            scaler = sklearn.preprocessing.MinMaxScaler().fit(np.concatenate([real_num, syn_num]))
            norm_real_num = scaler.transform(real_num)
            norm_syn_num = scaler.transform(syn_num)

            # compute the numerical distance matrix
            num_dist_matrix = ot.dist(norm_real_num, norm_syn_num, metric="minkowski", p=1)

            # compute the 2D wasserstein distance with linear programming
            # cost = 1(cat_real == cat_syn) + |num_real - num_syn|
            cost_matrix = cat_dist_matrix + num_dist_matrix
            wasserstein_error.append(ot.emd2([], [], cost_matrix))  # no need to assign weights

    return np.nanmean(wasserstein_error) if wasserstein_error else None


def num_num_error(real_data, syn_data):
    """
    compute the numerical error (2-way) for numerical columns between real and syn data
    """
    # for each combination of continuous columns, compute the 2D wasserstein distance
    wasserstein_error = []
    for i in range(real_data.shape[1] - 1):
        for j in range(i + 1, real_data.shape[1]):
            real_col1 = real_data[:, i].reshape(-1, 1)
            real_col2 = real_data[:, j].reshape(-1, 1)
            syn_col1 = syn_data[:, i].reshape(-1, 1)
            syn_col2 = syn_data[:, j].reshape(-1, 1)
            # normalize the column to [0, 1]
            scaler1 = sklearn.preprocessing.MinMaxScaler().fit(np.concatenate([real_col1, syn_col1]))
            scaler2 = sklearn.preprocessing.MinMaxScaler().fit(np.concatenate([real_col2, syn_col2]))
            norm_real_col1 = scaler1.transform(real_col1).flatten()
            norm_syn_col1 = scaler1.transform(syn_col1).flatten()
            norm_real_col2 = scaler2.transform(real_col2).flatten()
            norm_syn_col2 = scaler2.transform(syn_col2).flatten()
            # compute the 2D wasserstein distance with linear programming
            # concatenate the two columns
            real_col = np.concatenate([norm_real_col1.reshape(-1, 1), norm_real_col2.reshape(-1, 1)], axis=1)
            syn_col = np.concatenate([norm_syn_col1.reshape(-1, 1), norm_syn_col2.reshape(-1, 1)], axis=1)
            # use 1-norm as the distance metric
            cost_matrix = ot.dist(real_col, syn_col, metric="minkowski", p=1)
            wasserstein_error.append(ot.emd2([], [], cost_matrix))  # no need to assign weights
    return np.nanmean(wasserstein_error) if wasserstein_error else None


def cat_cat_error(real_data, syn_data):
    """
    compute the contigency error (2-way) for discrete columns between real and syn data
    """
    contigency_error = []
    for i in range(real_data.shape[1] - 1):
        for j in range(i + 1, real_data.shape[1]):
            marginal_diff = marginal_query(real_data.astype(str), syn_data.astype(str), [i,j], dimension=2)
            contigency_error.append(marginal_diff * 0.5)
    return np.nanmean(contigency_error) if contigency_error else None


###########################################################################
#########################1-way Wasserstein distance########################
###########################################################################


def num_error(real_data, syn_data):
    """
    compute the categorical error (1-way) for numerical columns between real and syn data
    """
    for i in range(real_data.shape[1]):
        wasserstein_error = []

        real_col = real_data[:,i].reshape(-1, 1)
        syn_col = syn_data[:,i].reshape(-1, 1)

        scaler = sklearn.preprocessing.MinMaxScaler().fit(np.concatenate([real_col, syn_col]))
        norm_real_col = scaler.transform(real_col).flatten()
        norm_syn_col = scaler.transform(syn_col).flatten()
        wasserstein_error.append(wasserstein_distance(norm_real_col, norm_syn_col))

    return np.nanmean(wasserstein_error) if wasserstein_error else None


def cat_error(real_data, syn_data):
    """
    compute the marginal error (1-way) for discrete columns between real and syn data
    """

    marginal_error = []
    for i in range(real_data.shape[1]):
        marginal_diff = marginal_query(real_data.astype(str), syn_data.astype(str), i)
        
        marginal_error.append(marginal_diff * 0.5)
    
    return np.nanmean(marginal_error) if marginal_error else None


def marginal_query(real_data, syn_data, col, dimension = 1):
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
        assert sum(syn_probs) >= 1 - 1e-2
    except:
        print("error in marginal_query for cols: ", col)
        print("real_probs: ", sum(real_probs))
        print("syn_probs: ", sum(syn_probs))
        raise ValueError("sum of probs should be 1")
    abs_diff = np.abs(np.array(real_probs) - np.array(syn_probs))
    return sum(abs_diff)


def make_fid(
    synthetic_data_path,
    data_path,
    task_type
):
    X_num_real, X_cat_real, y_real = read_pure_data(data_path, split = 'test')
    X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, split = 'train') 

    if task_type == 'regression':
        X_num_real = np.concatenate((X_num_real, y_real.reshape(-1,1)), axis=1)
        X_num_fake = np.concatenate((X_num_fake, y_fake.reshape(-1,1)), axis=1)
    else:
        X_cat_real = np.concatenate((X_cat_real, y_real.reshape(-1,1)), axis=1)
        X_cat_fake = np.concatenate((X_cat_fake, y_fake.reshape(-1,1)), axis=1)
    
    return cal_fidelity(X_num_real, X_cat_real, X_num_fake, X_cat_fake)


def eval_fid(
    raw_config,
    dataset = None,
    device = 'cuda:0',
    n_datasets = 5,
    sampling_method = 'ddpm',
    merf_dict = None,
    merf_rare_dict = None,
    privsyn_method = None,
    privsyn_preprocess = None
): 
    warnings.filterwarnings("ignore")
    
    parent_dir = Path(raw_config["parent_dir"])
    temp_config = deepcopy(raw_config)
    info = load_json(os.path.join(raw_config['real_data_path'], 'info.json'))
    task_type = info['task_type']
    ds = raw_config['real_data_path'].split('/')[-2]
    fid_res = {}
    
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        temp_config["parent_dir"] = str(dir_)
        if sampling_method == "merf":
            shutil.copy2(parent_dir / "merf_model.pt", temp_config["parent_dir"])
        elif sampling_method == 'privsyn':
            shutil.copy2(f'privsyn/temp_data/processed_data/{ds}_mapping', temp_config["parent_dir"])
        elif sampling_method == "ddpm":
            shutil.copy2(parent_dir / "model.pt", temp_config["parent_dir"])
        else: #dp_model
            shutil.copy2(parent_dir / "model.pt", temp_config["parent_dir"])

        for sample_seed in range(n_datasets):
            temp_config['sample']['seed'] = sample_seed
            dump_config(temp_config, dir_ / "config.toml")

            # sample via merf model
            if sampling_method == 'merf':
                merf_heterogeneous_sample(
                    **merf_dict,
                    parent_dir = temp_config['parent_dir'],
                    device = device,
                    cat_rare_dict = merf_rare_dict
                )
            
            # sample via privsyn
            elif sampling_method == 'privsyn':
                privsyn_method.synthesize_records()
                privsyn_method.postprocessing(dir_) # save raw synthesized data to temp dir

                privsyn_preprocess.reverse_mapping_from_files(temp_config['synthesized_filename'], temp_config['mapping_filename'], str(dir_)) 
                privsyn_preprocess.save_data_npy(dir_) # save reverse preprocessed synthesized data into temp dir
            
            # sample via dp_ddpm
            else:
                sample(
                    num_samples=temp_config['sample']['num_samples'],
                    batch_size=temp_config['sample']['batch_size'],
                    disbalance=temp_config['sample'].get('disbalance', None),
                    **temp_config['diffusion_params'],
                    parent_dir=temp_config['parent_dir'],
                    dataset = dataset,
                    data_path = temp_config['real_data_path'],
                    model_path=os.path.join(temp_config['parent_dir'], f'model.pt'),
                    model_type=temp_config['model_type'],
                    model_params=temp_config['model_params'],
                    T_dict=temp_config['train']['T'],
                    num_numerical_features=temp_config['num_numerical_features'],
                    device=device,
                    seed=temp_config['sample'].get('seed', 0),
                    dp = (sampling_method != 'ddpm')
                ) 

            synthetic_data_path = temp_config['parent_dir']
            data_path = temp_config['real_data_path']
            
            if not fid_res:
                for k,v in make_fid(synthetic_data_path, data_path, task_type).items():
                    fid_res[k] = [v]
            else:
                for k,v in make_fid(synthetic_data_path, data_path, task_type).items():
                    fid_res[k].append(v)

            print(f'Finish fidelity evaluation round {sample_seed}/{n_datasets}')
        
        shutil.rmtree(dir_)
    
    fid_report = {}
    for k,v in fid_res.items():
        fid_report[k] = {}
        fid_report[k]['mean'] = np.mean(fid_res[k])
        fid_report[k]['std'] = np.std(fid_res[k]) 

    print(fid_report)
    dump_json(fid_report, os.path.join(parent_dir, 'eval_fid.json'))

            


            

            