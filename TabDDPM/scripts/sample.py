import torch
import sys
target_path="./"
sys.path.append(target_path)

import numpy as np
import random
import os
from TabDDPM.model.diffusion import GaussianMultinomialDiffusion
from TabDDPM.model.modules import MLPDiffusion
from TabDDPM.data.dataset import make_dataset, Transformations
from scipy.spatial.distance import cdist
from opacus.grad_sample.grad_sample_module import GradSampleModule

class FoundNANsError(BaseException):
    """Found NANs during sampling"""
    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)

def get_model(
        model_params
):
    return MLPDiffusion(**model_params)

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


class ddpm_sampler():
    def __init__(self, diffusion, num_numerical_features, T_dict, dataset, model_params):
        self.diffusion = diffusion
        self.num_numerical_features_ = num_numerical_features
        self.T_dict = T_dict
        self.D = dataset
        self.model_params = model_params

    def sample(
        self,
        num_sample = 0,
        preprocesser = None,
        device = 'cuda:0',
        parent_dir = None,
        batch_size = 2000,
        disbalance = None,
        seed = 0
    ):
        np.random.seed(seed)

        X_num = None 
        X_cat = None

        device = torch.device(device)
        self.diffusion.to(device)
        self.diffusion.eval()
        
        print('Sampling ... ')
        _, empirical_class_dist = torch.unique(torch.from_numpy(self.D.y['train']), return_counts=True)
        
        if disbalance == 'fix':
            empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
            x_gen, y_gen = self.diffusion.sample_all(num_sample, batch_size, empirical_class_dist.float(), ddim=False)

        elif disbalance == 'fill':
            ix_major = empirical_class_dist.argmax().item()
            val_major = empirical_class_dist[ix_major].item()
            x_gen, y_gen = [], []
            for i in range(empirical_class_dist.shape[0]):
                if i == ix_major:
                    continue
                distrib = torch.zeros_like(empirical_class_dist)
                distrib[i] = 1
                num_sample = val_major - empirical_class_dist[i].item()
                x_temp, y_temp = self.diffusion.sample_all(num_sample, batch_size, distrib.float(), ddim=False)
                x_gen.append(x_temp)
                y_gen.append(y_temp)
            
            x_gen = torch.cat(x_gen, dim=0)
            y_gen = torch.cat(y_gen, dim=0)

        else:
            x_gen, y_gen = self.diffusion.sample_all(num_sample, batch_size, empirical_class_dist.float(), ddim=False)

        X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

        num_numerical_features = self.num_numerical_features_ + int(self.D.is_regression and not self.model_params["is_y_cond"])

        X_num_ = X_gen
        if num_numerical_features < X_gen.shape[1]:
            # np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
            if self.T_dict['cat_encoding'] == 'one-hot':
                X_gen[:, num_numerical_features:] = to_good_ohe(self.D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
            X_cat = self.D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

        if self.num_numerical_features_ != 0:
            # np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
            X_num_ = self.D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
            X_num = X_num_[:, :num_numerical_features]
            if self.model_params['num_classes'] == 0:
                y_gen = X_num[:, 0]
                X_num = X_num[:, 1:]

        # if num_numerical_features != 0:
        #     np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
        # if num_numerical_features < X_gen.shape[1]:
        #     np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
        # np.save(os.path.join(parent_dir, 'y_train'), y_gen)

        if num_numerical_features == 0: 
            df = np.hstack([X_cat, y_gen.reshape(-1,1)])
        elif num_numerical_features == X_gen.shape[1]:
            df = np.hstack([X_num, y_gen.reshape(-1,1)])
        else:
            df = np.hstack([X_num, X_cat, y_gen.reshape(-1,1)])

        preprocesser.reverse_data(df, parent_dir)
        print('Sample finished, store path:', parent_dir)
        torch.cuda.empty_cache()




# ============================= sample pretrain model, not used in the experiment ======================================
'''
def sample(
    parent_dir,
    dataset = None,
    batch_size = 2000,
    num_sample = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False,
    dp = True
):
    random.seed(seed)
    D = dataset

    cat_size = np.array(D.get_category_sizes('train'))
    if len(cat_size) == 0 or T_dict['cat_encoding'] == 'one-hot':
        cat_size = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(cat_size) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    if model_type == 'mlp':
        model = get_model(model_params).to(device)
    else: 
        raise ValueError('Not a mlp model')

    if dp:
        model = GradSampleModule(model)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        num_classes=cat_size,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, 
        num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, 
        scheduler=scheduler, 
        device=device
    )

    diffusion.to(device)
    diffusion.eval()
    
    print('Sampling ... ')
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_sample, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_sample = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_sample, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_sample, batch_size, empirical_class_dist.float(), ddim=False)

    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]

        # X_num_real = np.load(os.path.join(data_path, "X_num_train.npy"), allow_pickle=True)
        # disc_cols = []
        # for col in range(X_num_real.shape[1]):
        #     uniq_vals = np.unique(X_num_real[:, col])
        #     if len(uniq_vals) <= 16 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
        #         disc_cols.append(col)
        # if len(disc_cols) != 0:
        #     print("Discrete cols:", disc_cols)
        # if model_params['num_classes'] == 0:
        #     y_gen = X_num[:, 0]
        #     X_num = X_num[:, 1:]
        # if len(disc_cols):
        #     X_num = round_columns(X_num_real, X_num, disc_cols)

    # if model_params['num_classes'] != 0:
    #     y_gen = dataset.y_info['encoder'].reverse_transform(y_gen.reshape(-1,1)).squeeze()

    if num_numerical_features != 0:
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)
    print('Sample finished, store path:', parent_dir)

    del model 
    del diffusion
    torch.cuda.empty_cache()


def presample(
    parent_dir,
    dataset = None,
    batch_size = 2000,
    num_sample = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False
):
    random.seed(seed)
    D = dataset

    cat_size = np.array(D.get_category_sizes('train'))
    if len(cat_size) == 0 or T_dict['cat_encoding'] == 'one-hot':
        cat_size = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(cat_size) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    if model_type == 'mlp':
        model = get_model(model_params).to(device)
    else: 
        raise ValueError('Not a mlp model')

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        num_classes=cat_size,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, 
        num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, 
        scheduler=scheduler, 
        device=device
    )

    diffusion.to(device)
    diffusion.eval()
    
    print('Start sampling ... ')
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_sample, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_sample = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_sample, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_sample, batch_size, empirical_class_dist.float(), ddim=False)

    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_pre_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_pre_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]

        
        X_num_real = np.load(os.path.join(data_path, "X_num_pretrain.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)
        

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_pretrain'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_pretrain'), X_cat)
    np.save(os.path.join(parent_dir, 'y_pretrain'), y_gen)
    print('Sample finished, store path:', parent_dir)
'''