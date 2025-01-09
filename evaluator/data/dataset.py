import sys
target_path="./"
sys.path.append(target_path)

import numpy as np
import pandas as pd
import torch
import os
import hashlib
import sklearn
import random
import warnings
import itertools
from torch.utils.data import DataLoader, TensorDataset
from TabDDPM.data.data_utils import *
from copy import deepcopy
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, astuple, replace
from category_encoders import LeaveOneOutEncoder
from preprocess_common.preprocess import * 
from TabDDPM.data.metrics import calculate_metrics
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from tqdm import tqdm


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)
        splits = [k for k in ['pretrain', 'preval', 'pretest', 'train', 'val', 'test'] if dir_.joinpath(f'y_{k}.npy').exists()]

        def load(item) -> ArrayDict:
            return {
                x: cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle=True))  # type: ignore[code]
                for x in splits
            }

        if Path(dir_ / 'info.json').exists():
            info = load_json(dir_ / 'info.json')
        else:
            info = None
        return Dataset(
            load('X_num') if dir_.joinpath('X_num_train.npy').exists() else None,
            load('X_cat') if dir_.joinpath('X_cat_train.npy').exists() else None,
            load('y'),
            {},
            TaskType(info['task_type']),
            info.get('n_classes'),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1
    
    def reverse_cat_rare(self, X_cat) -> np.ndarray:
        if (self.cat_rare_dict is None) or (all(len(sublist) == 0 for sublist in self.cat_rare_dict)):
            print('No rare categorical value')
            return X_cat 
        else: 
            for column_idx in range(len(self.cat_rare_dict)):
                idx = (X_cat[:, column_idx] == CAT_RARE_VALUE)
                if (len(self.cat_rare_dict[column_idx]) > 0) & (idx.any()):
                    X_cat[idx, column_idx] = np.random.choice(self.cat_rare_dict[column_idx], size=sum(idx), replace = True)
            return X_cat


    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = 'rmse'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]
        return metrics

    def syn_pretrain_data(self, epsilon, delta, sample_num, **kwargs):
        """
        This function is used to generate pretrain data from train data
        """
        import AIM.mbi.Dataset as aim_ds
        from AIM.aim import AIM 

        if self.X_num is None: 
            aim_data = aim_ds.Dataset.load_from_dataset(None, self.X_cat['train'], self.y['train'])
        elif self.X_cat is None:
            aim_data = aim_ds.Dataset.load_from_dataset(self.X_num['train'], None, self.y['train'])
        else:
            aim_data = aim_ds.Dataset.load_from_dataset(self.X_num['train'], self.X_cat['train'], self.y['train'])

        workload = list(itertools.combinations(aim_data.domain, kwargs.get("degree", 2)))
        workload = [cl for cl in workload if aim_data.domain.size(cl) <= kwargs.get("max_cells", 10000)]
        workload = [(cl, 1.0) for cl in workload]

        mech = AIM(
            epsilon,
            delta,
            max_model_size=kwargs.get("max_model_size", 80),
            max_iters=kwargs.get("max_iters", 1000),
        )
        mech.run(aim_data, workload)
        syn = mech.syn_data(
            num_synth_rows = sample_num
        )

        num_row = self.X_num['train'].shape[1] if self.X_num is not None else 0
        cat_row = self.X_cat['train'].shape[1] if self.X_cat is not None else 0

        if self.X_num is not None:
            self.X_num['pretrain'] = syn.df.iloc[:, 0: num_row].to_numpy().astype(float)
        if self.X_cat is not None: 
            self.X_cat['pretrain'] = syn.df.iloc[:, num_row: num_row + cat_row].to_numpy().astype(int) if self.X_cat is not None else None
        self.y['pretrain'] = syn.df.iloc[:, -1].to_numpy().reshape(-1).astype(int)

        return 0

    def syn_pretrain_data_merf(self, epsilon, delta, device, sample_num, **kwargs):
        from DP_MERF.single_generator_priv_all import merf_main
        from DP_MERF.sample import merf_heterogeneous_sample

        merf_model, merf_dict = merf_main(
                                    dataset = 'pretrain',
                                    dp_epsilon = epsilon,
                                    device=device,
                                    dp_delta = delta,
                                    seed_number=0,
                                    n_features_arg = 2000, 
                                    mini_batch_size_arg = 0.05, 
                                    how_many_epochs_arg = 500, 
                                    is_priv_arg = True,
                                    X_num = self.X_num['train'] if self.X_num is not None else None,
                                    X_cat = self.X_cat['train'] if self.X_cat is not None else None,
                                    y = self.y['train'],
                                    eval=False
                                )

        merf_dict['n'] = sample_num

        x_num, x_cat, y = merf_heterogeneous_sample(
            **merf_dict,
            parent_dir = None,
            device = device,
            model=merf_model, 
            save=False
        )

        if self.X_num is not None:
            self.X_num['pretrain'] = x_num.astype(float)
        if self.X_cat is not None: 
            self.X_cat['pretrain'] = x_cat.astype(int) if self.X_cat is not None else None
        self.y['pretrain'] = y.astype(int)

        return 0

    

    def pretrain_data_imputation(self, rho, seed = 0, margin_all = False):
        random.seed(seed)
        num_miss = []
        cat_miss = []
        self.num_margin = []
        self.cat_margin = []
        self.y_margin = {}
        
        if self.X_num is not None: 
            num_miss = list(range(self.X_num['pretrain'].shape[1]))
            num_miss_attr = len(num_miss) if margin_all else np.isnan(self.X_num['pretrain']).any(axis=0).sum() 
        if self.X_cat is not None:
            cat_miss = list(range(self.X_cat['pretrain'].shape[1]))
            cat_miss_attr = len(cat_miss) if margin_all else np.isnan(self.X_cat['pretrain']).any(axis=0).sum() 

        miss_attr = num_miss_attr + cat_miss_attr + int(margin_all)
        if miss_attr == 0:
            print('No missing data')
            return 0
        else:
            rho_attr = rho/miss_attr 

            for i in num_miss:
                element, count = np.unique(self.X_num['train'][:, i], return_counts = True)
                count = count + np.sqrt(2.0) * np.sqrt(1 / (2*rho_attr)) * np.random.randn(len(count))
                count = np.where(count < 0, 0.0, count)

                if margin_all:
                    self.num_margin.append({})
                    for j in range(len(element)):
                        self.num_margin[i][element[j]] = count[j]/sum(count)

                miss_idx = np.isnan(self.X_num['pretrain'][:, i])
                if miss_idx.any():
                    self.X_num['pretrain'][miss_idx, i] = np.random.choice(
                        element,
                        size = sum(miss_idx),
                        p = count/np.sum(count)
                    )
            for i in cat_miss:
                element, count = np.unique(self.X_cat['train'][:, i], return_counts = True)
                count = count + np.sqrt(2.0) * np.sqrt(1 / (2*rho_attr)) * np.random.randn(len(count))
                count = np.where(count < 0, 0.0, count)

                if margin_all:
                    self.cat_margin.append({})
                    for j in range(len(element)):
                        self.cat_margin[i][element[j]] = count[j]/sum(count)

                miss_idx = np.isnan(self.X_cat['pretrain'][:, i])
                if miss_idx.any():
                    self.X_cat['pretrain'][miss_idx, i] = np.random.choice(
                        element,
                        size = sum(miss_idx),
                        p = count/np.sum(count)
                    )
            if margin_all:
                element, count = np.unique(self.y['train'], return_counts = True)
                count = count + np.sqrt(2.0) * np.sqrt(1 / (2*rho_attr)) * np.random.randn(len(count))
                count = np.where(count < 0, 0.0, count)
                for j in range(len(element)):
                    self.y_margin[element[j]] = count[j]/sum(count)

            print('Finish pretrain data imputation')
            return 0
    

    def update_pretrain_data(self, idx_filter, aug_data = {}):
        if aug_data: 
            self.X_num['pretrain'] = np.concatenate((self.X_num['pretrain'][np.where(idx_filter == 1)[0]], aug_data['X_num']))
            self.X_cat['pretrain'] = np.concatenate((self.X_cat['pretrain'][np.where(idx_filter == 1)[0]], aug_data['X_cat']))
            self.y['pretrain'] = np.concatenate((self.y['pretrain'][np.where(idx_filter == 1)[0]], aug_data['y']))
        else: 
            self.X_num['pretrain'] = self.X_num['pretrain'][np.where(idx_filter == 1)[0]]
            self.X_cat['pretrain'] = self.X_cat['pretrain'][np.where(idx_filter == 1)[0]]
            self.y['pretrain'] = self.y['pretrain'][np.where(idx_filter == 1)[0]]
        return 0

    def subset_train_data(self):
        # this is a function for debug, dont use
        if self.X_num is not None:
            self.X_num['train'] = self.X_num['train'][0:2000, :]
        if self.X_cat is not None:
            self.X_cat['train'] = self.X_cat['train'][0:2000, :]
        self.y['train'] = self.y['train'][0:2000]
        return 0



def normalize(
        X: ArrayDict, 
        normalization: Normalization, 
        seed: Optional[int], 
        return_normalizer : bool = False
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler(
            feature_range=(-1, 1)
        )
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError('normalization:', normalization)
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}

def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    popular_categories = []
    impopupar_categories = []

    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories.append([k for k, v in counter.items() if v >= min_count])
        impopupar_categories.append([k for k, v in counter.items() if v < min_count])
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories[column_idx] else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}, impopupar_categories

def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder : bool = False,
    rho_cat = 0.0,
    rare_threshold = 0.005
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    if encoding is None:
        unknown_value = np.nan
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='float',  # type: ignore[code]
        ).fit(X['train'])

        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}

        for part in X.keys(): 
            # simply impute missing values in val and test data
            if part in ['pretrain', 'train']: continue
            for column_idx in range(X[part].shape[1]):
                unknown_list = np.isnan(X[part][:, column_idx]) 
                X[part][unknown_list, column_idx] = (
                    np.random.choice(X['train'][:, column_idx], size=sum(unknown_list), replace=True)
                )
    
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    elif encoding == 'default':
        encoder = rare_merger(
            rho_cat, 
            rare_threshold=rare_threshold,
            output_type = 'ordinal'
        )
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        return (X, False)
    
    elif encoding == 'one-hot':
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}

    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(('loe', loe))
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]

    else:
        raise ValueError('encoding:', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION: #only when task is regression, scale the y
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
        
        '''
        else: # if class task, ordinal encoder 
            encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='error',  # type: ignore[code]
                dtype='int',  # type: ignore[code]
            ).fit(y['train'].reshape(-1,1))
            y = {k: encoder.transform(v.reshape(-1,1)).squeeze() for k, v in y.items()}
            info['encoder'] = encoder
        '''
    else:
        raise ValueError('policy', policy)
    return y, info


def change_val_fn(dataset: Dataset, have_pretrain: int = 0, val_size: float = 0.2):
    # should be done before transformations

    y = np.concatenate([dataset.y['train'], dataset.y['val']], axis=0)
    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=0)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=0, stratify=y)

    dataset.y['train'] = y[train_ixs]
    dataset.y['val'] = y[val_ixs]

    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num['train'], dataset.X_num['val']], axis=0)
        dataset.X_num['train'] = X_num[train_ixs]
        dataset.X_num['val'] = X_num[val_ixs]

    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat['train'], dataset.X_cat['val']], axis=0)
        dataset.X_cat['train'] = X_cat[train_ixs]
        dataset.X_cat['val'] = X_cat[val_ixs]
    
    if have_pretrain:
        y_pre = np.concatenate([dataset.y['pretrain'], dataset.y['preval']], axis=0)
        pre_ixs = np.arange(y_pre.shape[0])

        if dataset.is_regression:
            pretrain_ixs, preval_ixs = train_test_split(pre_ixs, test_size=val_size, random_state=0)
        else:
            pretrain_ixs, preval_ixs = train_test_split(pre_ixs, test_size=val_size, random_state=0, stratify=y_pre)

        dataset.y['pretrain'] = y_pre[pretrain_ixs]
        dataset.y['preval'] = y_pre[preval_ixs]

        if dataset.X_num is not None:
            X_num_pre = np.concatenate([dataset.X_num['pretrain'], dataset.X_num['preval']], axis=0)
            dataset.X_num['pretrain'] = X_num_pre[pretrain_ixs]
            dataset.X_num['preval'] = X_num_pre[preval_ixs]

        if dataset.X_cat is not None:
            X_cat_pre = np.concatenate([dataset.X_cat['pretrain'], dataset.X_cat['preval']], axis=0)
            dataset.X_cat['pretrain'] = X_cat_pre[train_ixs]
            dataset.X_cat['preval'] = X_cat_pre[val_ixs]

    return dataset


def read_pure_data(path, split='train'):
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True)

    return X_num, X_cat, y

def read_changed_val(path, val_size=0.2, model_step='finetune', seed=0):
    path = Path(path)
    if model_step == 'finetune':
        X_num_train, X_cat_train, y_train = read_pure_data(path, 'train')
        X_num_val, X_cat_val, y_val = read_pure_data(path, 'val')
        is_regression = load_json(path / 'info.json')['task_type'] == 'regression'

        y = np.concatenate([y_train, y_val], axis=0)

        ixs = np.arange(y.shape[0])
        if is_regression:
            train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=seed)
        else:
            train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=seed, stratify=y)
        y_train = y[train_ixs]
        y_val = y[val_ixs]

        if X_num_train is not None:
            X_num = np.concatenate([X_num_train, X_num_val], axis=0)
            X_num_train = X_num[train_ixs]
            X_num_val = X_num[val_ixs]

        if X_cat_train is not None:
            X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
            X_cat_train = X_cat[train_ixs]
            X_cat_val = X_cat[val_ixs]
        
        return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val
    else: 
        X_num_train, X_cat_train, y_train = read_pure_data(path, 'pretrain')
        X_num_val, X_cat_val, y_val = read_pure_data(path, 'preval')
        is_regression = load_json(path / 'info.json')['task_type'] == 'regression'

        y = np.concatenate([y_train, y_val], axis=0)

        ixs = np.arange(y.shape[0])
        if is_regression:
            train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=seed)
        else:
            train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=seed, stratify=y)
        y_train = y[train_ixs]
        y_val = y[val_ixs]

        if X_num_train is not None:
            X_num = np.concatenate([X_num_train, X_num_val], axis=0)
            X_num_train = X_num[train_ixs]
            X_num_val = X_num[val_ixs]

        if X_cat_train is not None:
            X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
            X_cat_train = X_cat[train_ixs]
            X_cat_val = X_cat[val_ixs]
        
        return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    return_transforms: bool = False,
    rho_cat=0.0, 
    rare_threshold=0.0
) -> Dataset:
    # This function 
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    # if dataset.X_num is not None:
        # dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True
        )
        num_transform = num_transform


    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        # X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        X_cat = dataset.X_cat
        
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y['train'],
            transformations.seed,
            return_encoder=True,
            rho_cat = rho_cat,
            rare_threshold = rare_threshold
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None 
        

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        dump_pickle((transformations, dataset), cache_path)
    if return_transforms:
        return dataset, num_transform, cat_transform
    else:
        return dataset


def make_dataset(
    data_path: str,
    T: Transformations, #transformation contains the policies of normalization, encoding and nan value process
    num_classes: int,
    is_y_cond: bool,
    change_val: bool,
    have_pretrain = 0,
    y_num_classes = None,
    task_type = 'binclass',
    rho = 1.0,
    rare_threshold = 0.005
):  
    split_set = ['pretrain', 'preval', 'pretest', 'train', 'val', 'test'] if have_pretrain else ['train', 'val', 'test']

    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in split_set:
            if Path(data_path).joinpath(f'y_{split}.npy').exists():
                X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
                if X_num is not None:
                    X_num[split] = X_num_t
                if not is_y_cond:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                if X_cat is not None:
                    X_cat[split] = X_cat_t
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in split_set:
            if Path(data_path).joinpath(f'y_{split}.npy').exists():
                X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
                if not is_y_cond:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                if X_num is not None:
                    X_num[split] = X_num_t
                if X_cat is not None:
                    X_cat[split] = X_cat_t
                y[split] = y_t

    D = Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type = TaskType(task_type),
        n_classes=y_num_classes
    )

    _, rho_cat = calculate_rho_allocate(0.1*rho, None, D.X_cat['train'], None)

    D = transform_dataset(D, T, None, False, rho_cat, rare_threshold)
    if change_val:
        D = change_val_fn(D, have_pretrain) #generate new train, validation data from raw dataset
    D.y_info['num_classes'] = D.n_classes
    return D, rho_cat


def make_dataset_from_df(
        df, 
        T,
        y_num_classes: int,   
        is_y_cond: bool,
        task_type: str
    ):
    if not is_y_cond:
        df['X_num'] = np.concatenate([df['y'].reshape(-1, 1), df['X_num']], axis=1)

    D = Dataset(
        {'train': df['X_num']} if df['X_num'] is not None else None,
        {'train': df['X_cat']} if df['X_cat'] is not None else None,
        {'train': df['y']},
        y_info = {},
        task_type = TaskType(task_type),
        n_classes = y_num_classes
    )
    D = transform_dataset(D, T, None, False, None, None)
    D.y_info['num_classes'] = D.n_classes
    return D



class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def prepare_torch_dataloader(
    D: Dataset,
    split: str,
    batch_size: int
): 
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.tensor(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1), dtype=torch.float32)
        else:
            X = torch.tensor(D.X_cat[split], dtype=torch.float32)
    else:
        X = torch.tensor(D.X_num[split], dtype=torch.float32)
    y = torch.tensor(D.y[split])
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split in ['pretrain', 'train']))



def prepare_fast_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split in ['pretrain', 'train']))
    while True:
        yield from dataloader

def concat_features(D : Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()}
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat[part],
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X

def update_pretrain_data(dataset: Dataset, idx: np.array):
    dataset.X_num['pretrain'] = dataset.X_num['pretrain'][np.where(idx == 1)[0]]
    dataset.X_cat['pretrain'] = dataset.X_cat['pretrain'][np.where(idx == 1)[0]]
    dataset.y['pretrain'] = dataset.y['pretrain'][np.where(idx == 1)[0]]

    return dataset

def prepare_datapreprocess_dataloader(dataset, split, batch_size, num_size = 0, cat_size = [0], y_size = 1, return_dataloader = True):
    num_part = None
    cat_part = None
    if num_size != 0:
        num_part = torch.tensor(dataset.X_num[split], dtype = torch.float32)
    if np.sum(cat_size) != 0:
        cat_part = torch.tensor(dataset.X_cat[split], dtype = torch.int64)
    if y_size == 1:
        y_part = torch.tensor(dataset.y[split], dtype = torch.float32).reshape(-1,1)
    else:
        y_part = torch.tensor(dataset.y[split], dtype = torch.int64).reshape(-1,1)
    
    if return_dataloader == False:
        return num_part, cat_part, y_part
    else:
        if num_size == 0:
            ds = TensorDataset(cat_part, y_part)
        elif np.sum(cat_size) == 0:
            ds = TensorDataset(num_part, y_part)
        else:
            ds = TensorDataset(num_part, cat_part, y_part)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)



class BlockRandomFourierFeatureProcesser:
    def __init__(self, dataset: Dataset, feature_dim, rho, block_size=512, seed=0):
        np.random.seed(seed)
        self.feature_dim = feature_dim 
        self.rho = rho
        self.seed = seed

        self.pretrain_m = dataset.y['pretrain'].shape[0]
        self.train_m = dataset.y['train'].shape[0]
        self.y_num_classes = dataset.y_info['num_classes'] if dataset.y_info['num_classes'] is not None else 1  # category number of y
        # self.Delta = 2 /self.train_m
        self.Delta = 2 * np.sqrt(2) / self.train_m
        self.sigma = np.sqrt(1/(2 * self.rho))

        if dataset.X_num is not None:
            self.num_num_classes = dataset.X_num['train'].shape[1]
        else:
            self.num_num_classes = None
            self.Delta = 2/self.train_m
        
        if dataset.X_cat is not None:
            self.cat_num_classes = np.array(dataset.get_category_sizes('train'))
        else:
            self.cat_num_classes = [0]
            self.Delta = 2/self.train_m
        
        if self.y_num_classes == None:
            self.y_type = 'regression'
        else: 
            self.y_type = 'class'
        
        for part in ['pretrain', 'train']:
            if dataset.X_cat is not None:
                dataset.X_cat[part] = index_to_onehot(dataset.X_cat[part], self.cat_num_classes)
        
            if self.y_type == 'class':
                dataset.y[part] = index_to_onehot(dataset.y[part], np.array([self.y_num_classes]))
            elif self.y_type == 'regression':
                dataset.X_num[part] = np.hstack(dataset.y[part], dataset.X_num[part]) # if regression, merge X_num with y
        self.attr_num = self.num_num_classes + len(self.cat_num_classes) + 1

        self.dataset = dataset
        if self.dataset.X_num is not None: 
            self.w = np.random.randn(self.feature_dim // 2, self.dataset.X_num['train'].shape[1])
            self.w = np.vstack((self.w, self.w.copy()))

        self.filter_idx = np.ones(len(dataset.y['pretrain']))
        self.aug_idx = []
        self.block_idx = generate_shuffle_block(np.arange(len(dataset.y['pretrain'])), block_size)
        print("-"*100)
        print(f'block size: {block_size}, number of block: {len(self.block_idx)}, numerical feature dimension: {self.feature_dim}')


    def RFF(self, x:np.array, feature_type):
        if feature_type == 'num':
            phi = np.matmul(x, self.w.T)
            if x.ndim == 2: 
                cos_half = np.cos(phi[: ,: self.feature_dim // 2])
                sin_half = np.sin(phi[: ,self.feature_dim // 2 :])
                phi[:, : self.feature_dim // 2] = cos_half
                phi[:, self.feature_dim // 2 :] = sin_half
            elif x.ndim == 1: 
                cos_half = np.cos(phi[: self.feature_dim // 2])
                sin_half = np.sin(phi[self.feature_dim // 2 :])
                phi[: self.feature_dim // 2] = cos_half
                phi[self.feature_dim // 2 :] = sin_half
        
            return np.sqrt(2/self.feature_dim) * phi
        
        if feature_type == 'cat':
            return x / np.sqrt(len(self.cat_num_classes))

        if feature_type == 'y': 
            return x

        else: 
            raise ValueError('Invalid kernel type')

    def clip(self, x:np.array):
        """
        This function is used for clipping RFF, to make better convergence. 

        prove to be useless :)
        """
        x[:self.feature_dim] = np.clip(x[:self.feature_dim], -np.sqrt(2/self.feature_dim), np.sqrt(2/self.feature_dim))
        x[self.feature_dim:] = np.clip(x[self.feature_dim:], 0, np.sqrt(1/len(self.cat_num_classes)))
        return x 

    def initialize_fourier_feature(self, clip = False):
        np.random.seed(self.seed)
        
        if self.y_type == 'class':
            # RFF of pretrain data
            if self.dataset.X_num is None:
                fx = self.RFF(self.dataset.X_cat['pretrain'], 'cat')
            elif self.dataset.X_cat is None:
                fx = self.RFF(self.dataset.X_num['pretrain'], 'num')
            else:
                fx = np.hstack((self.RFF(self.dataset.X_num['pretrain'], 'num'), self.RFF(self.dataset.X_cat['pretrain'], 'cat')))
            fy = self.RFF(self.dataset.y['pretrain'], 'y')

            self.pretrain_RFF_list = np.zeros((self.pretrain_m, self.feature_dim + np.sum(self.cat_num_classes), self.y_num_classes))
            for i in range(self.pretrain_m):
                self.pretrain_RFF_list[i] = np.outer(fx[i], fy[i])
            self.pretrain_RFF = [np.mean(self.pretrain_RFF_list[x], axis=0) for x in self.block_idx]
            self.pretrain_RFF_total = np.mean(self.pretrain_RFF_list, axis = 0)


            # differential private RFF of pretrain data
            if self.dataset.X_num is None:
                fx = self.RFF(self.dataset.X_cat['train'], 'cat')
            elif self.dataset.X_cat is None:
                fx = self.RFF(self.dataset.X_num['train'], 'num')
            else:
                fx = np.hstack((self.RFF(self.dataset.X_num['train'], 'num'), self.RFF(self.dataset.X_cat['train'], 'cat')))
            fy = self.RFF(self.dataset.y['train'], 'y')

            self.train_RFF = np.zeros((self.feature_dim + np.sum(self.cat_num_classes), self.y_num_classes))
            for i in range(self.train_m):
                self.train_RFF += np.outer(fx[i], fy[i]) / self.train_m 

            RFF_noise = self.Delta * self.sigma * np.random.randn(self.feature_dim + np.sum(self.cat_num_classes), self.y_num_classes)
            self.train_RFF += RFF_noise
            if clip: self.train_RFF = self.clip(self.train_RFF)
        
        if self.y_type == 'regression':
            # RFF of pretrain data
            if self.dataset.X_num is None:
                fx = self.RFF(self.dataset.X_cat['pretrain'], 'cat')
            elif self.dataset.X_cat is None:
                fx = self.RFF(self.dataset.X_num['pretrain'], 'num')
            else:
                fx = np.hstack((self.RFF(self.dataset.X_num['pretrain'], 'num'), self.RFF(self.dataset.X_cat['pretrain'], 'cat')))
            self.pretrain_RFF_list = np.zeros((self.pretrain_m, self.feature_dim + np.sum(self.cat_num_classes)))
            for i in range(self.pretrain_m):
                self.pretrain_RFF_list[i] = fx[i]
            self.pretrain_RFF = [np.mean(self.pretrain_RFF_list[x], axis=0) for x in self.block_idx]

            # differential private RFF of pretrain data
            if self.dataset.X_num is None:
                fx = self.RFF(self.dataset.X_cat['train'], 'cat')
            elif self.dataset.X_cat is None:
                fx = self.RFF(self.dataset.X_num['train'], 'num')
            else:
                fx = np.hstack((self.RFF(self.dataset.X_num['train'], 'num'), self.RFF(self.dataset.X_cat['train'], 'cat')))
            self.train_RFF = np.mean(fx, axis=0)

            RFF_noise = self.Delta * self.sigma * np.random.randn(self.feature_dim + np.sum(self.cat_num_classes))
            self.train_RFF += RFF_noise 
            if clip: self.train_RFF = self.clip(self.train_RFF)
            
        print("Finish initializing RFF")

    def update_pretrain_RFF_delete(self, block_filter_idx, j, i):
        if sum(block_filter_idx) > 1:
            RFF = sum(block_filter_idx) * self.pretrain_RFF[j] - self.pretrain_RFF_list[self.block_idx[j][i]] 
            RFF = RFF/(sum(block_filter_idx) - 1) 
            return RFF
        else:
            if self.y_type == 'class':
                return np.zeros((self.feature_dim + np.sum(self.cat_num_classes), self.y_num_classes))
            elif self.y_type == 'regression':
                return np.zeros((self.feature_dim + np.sum(self.cat_num_classes) + 1)) 

    def RFF_greedy_filter_process(self, save_path, threshold_param):
        np.random.seed(self.seed)
        record_df = pd.DataFrame(columns = ['block', 'start_MMD', 'end_MMD', 'start_data_num', 'end_data_num'])
        if self.y_type == 'class':
            with tqdm(total = len(self.block_idx)) as pbar:
                # iteration for each block
                for j in range(len(self.block_idx)):
                    block_filter_idx = np.ones(len(self.block_idx[j]))
                    MMD = np.linalg.norm(self.pretrain_RFF[j] - self.train_RFF, ord='fro')**2
                    count = 0
                    record = [j+1, deepcopy(MMD), 0, len(block_filter_idx), 0]

                    while sum(block_filter_idx) > 0:
                        best_improvement = 0.0
                        threshold = threshold_param * MMD / sum(block_filter_idx)
                        for i in np.where(block_filter_idx == 1)[0]:
                            temp_RFF = self.update_pretrain_RFF_delete(block_filter_idx, j, i)
                            temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='fro')**2
                            temp_improvement = MMD - temp_MMD
                            if temp_improvement >= best_improvement:
                                best_improvement = temp_improvement 
                                best_RFF = temp_RFF 
                                best_MMD = temp_MMD
                                best_i = i

                        if best_improvement > threshold:
                            block_filter_idx[best_i] = 0
                            self.filter_idx[self.block_idx[j][best_i]] = 0
                            self.pretrain_RFF[j] = best_RFF
                            MMD = best_MMD
                            count += 1
                        else: 
                            break

                    record[2] = MMD
                    record[4] = sum(block_filter_idx)
                    record_df.loc[j] = record

                    pbar.update(1)
                    pbar.set_description(f"Filter")
                    pbar.set_postfix(Filter_info = f'{count}/{len(block_filter_idx)}') 

        elif self.y_type == 'regression':
            with tqdm(total = len(self.block_idx)) as pbar:
                for j in range(len(self.block_idx)):
                    block_filter_idx = np.ones(len(self.block_idx[j]))
                    MMD = np.linalg.norm(self.pretrain_RFF[j] - self.train_RFF, ord='2')**2
                    count = 0
                    record = [j+1, deepcopy(MMD), 0, len(block_filter_idx), 0]
                    while sum(block_filter_idx) > 0: # minimum sample number for accuracy 
                        best_improvement = 0.0
                        threshold = threshold_param * MMD / sum(block_filter_idx) # dynamic threshold for data filter
                        for i in np.where(block_filter_idx == 1)[0]:
                            temp_RFF = self.update_pretrain_RFF_delete(block_filter_idx, j, i)
                            temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='2')**2
                            temp_improvement = MMD - temp_MMD
                            if temp_improvement >= best_improvement:
                                best_improvement = temp_improvement 
                                best_RFF = temp_RFF 
                                best_MMD = temp_MMD
                                best_i = i

                        if best_improvement > threshold:
                            block_filter_idx[best_i] = 0
                            self.filter_idx[self.block_idx[j][best_i]] = 0
                            self.pretrain_RFF[j] = best_RFF
                            MMD = best_MMD
                            count += 1
                        else: 
                            break
                    
                    record[2] = MMD
                    record[4] = sum(block_filter_idx)
                    record_df.loc[j] = record

                    pbar.update(1)
                    pbar.set_description(f"Filter")
                    pbar.set_postfix(Filter_info = f'{count}/{len(block_filter_idx)}')
        
        record_df.loc[len(self.block_idx)] = ['all', '-', '-', len(self.filter_idx), sum(self.filter_idx)]
        record_df['start_data_num'] = record_df['start_data_num'].astype(int)
        record_df['end_data_num'] = record_df['end_data_num'].astype(int)
        record_df.to_csv(os.path.join(save_path, 'preprocess_record.csv'), index = False)

        print(f'Totally {len(self.filter_idx) - sum(self.filter_idx)} ({100 * (1 - sum(self.filter_idx)/len(self.filter_idx)) :.2f}%) data has been removed, {sum(self.filter_idx)} ({100 * sum(self.filter_idx)/len(self.filter_idx) :.2f}%) data remained')
        if sum(self.filter_idx) <= 0.05 * len(self.filter_idx):
            warnings.warn('Too much data has been removed. Please try larger threshold param or other pretrain dataset', UserWarning)
        print("-"*100)
        return self.filter_idx 



    def RFF_genetic_aug_process(self):
        current_activate_idx = np.where(self.filter_idx == 1)[0]
        pretrain_m_after_filter = len(current_activate_idx) # current number of pretrain data 
        pretrain_RFF_all = np.mean([self.pretrain_RFF_list[x] for x in current_activate_idx], axis=0) # current RFF of pretrain dataset after filter
        MMD = np.linalg.norm(pretrain_RFF_all - self.train_RFF, ord='fro')**2 # current MMD
        start_MMD = deepcopy(MMD)

        aug_num_iteration = int((self.dataset.y['train'].shape[0] - pretrain_m_after_filter)/20) # how many number of augmentation
        aug_change_attr = max(int(0.1 * self.attr_num), 1)
        self.aug_data_all = {}
        aug_data_all_length = 0

        if aug_num_iteration < 1:
            return self.aug_data_all

        print(f'Number of augment iteration is 20, augment data number each iteration is {aug_num_iteration}, attributes changed each iteration is {aug_change_attr}')
        if self.y_type == 'class':
            num_range = get_numerical_range(self.dataset.X_num['train']) if self.dataset.X_num is not None else 0
            cat_range = [0] + list(itertools.accumulate(self.cat_num_classes)) if self.dataset.X_cat is not None else []
            y_range = [0, self.dataset.y['train'].shape[1]]
            current_m_after_filter = len(current_activate_idx) # current how many data is included in the pretrain data

            for _ in tqdm(range(20), desc='Augmentation'):
                iter = 0
                temp_data = {}
                while iter < aug_num_iteration:
                    imp = 0.0
                    while imp < 0.1 * MMD/current_m_after_filter:
                        idx = random.randint(0, pretrain_m_after_filter + aug_data_all_length - 1) # random select a value
                        if idx < pretrain_m_after_filter:
                            aug_data_num_origin = self.dataset.X_num['pretrain'][current_activate_idx[idx]] if self.dataset.X_num is not None else None 
                            aug_data_cat_origin = self.dataset.X_cat['pretrain'][current_activate_idx[idx]] if self.dataset.X_cat is not None else None 
                            aug_data_y_origin = self.dataset.y['pretrain'][current_activate_idx[idx]]
                        else:
                            aug_data_num_origin = self.aug_data_all['X_num'][idx - pretrain_m_after_filter] if self.dataset.X_num is not None else None 
                            aug_data_cat_origin = self.aug_data_all['X_cat'][idx - pretrain_m_after_filter] if self.dataset.X_cat is not None else None 
                            aug_data_y_origin = self.aug_data_all['y'][idx - pretrain_m_after_filter] 
                        
                        aug_data_num, aug_data_cat, aug_data_y = self.change_aug(
                                                                    deepcopy(aug_data_num_origin), 
                                                                    deepcopy(aug_data_cat_origin), 
                                                                    deepcopy(aug_data_y_origin), 
                                                                    num_range, cat_range, y_range, aug_change_attr
                                                                )
                        assert (
                            not np.array_equal(aug_data_num, aug_data_num_origin) or 
                            not np.array_equal(aug_data_cat, aug_data_cat_origin) or 
                            not np.array_equal(aug_data_y, aug_data_y_origin)
                        ), 'Invalid data change'

                        if aug_data_num is None:
                            fx = self.RFF(aug_data_cat, 'cat')
                        elif aug_data_cat is None:
                            fx = self.RFF(aug_data_num, 'num')
                        else:
                            fx = np.concatenate((self.RFF(aug_data_num, 'num'), self.RFF(aug_data_cat, 'cat')))
                        fy = self.RFF(aug_data_y, 'y')
                        new_RFF = np.outer(fx, fy) 

                        temp_RFF = (new_RFF + pretrain_RFF_all * current_m_after_filter)/(current_m_after_filter + 1)
                        temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='fro')**2
                        imp = MMD - temp_MMD

                    MMD = temp_MMD
                    pretrain_RFF_all = temp_RFF 
                    current_m_after_filter += 1 
                    iter += 1

                    if temp_data:
                        temp_data['X_num'] = np.vstack((temp_data['X_num'], np.array([aug_data_num])))
                        temp_data['X_cat'] = np.vstack((temp_data['X_cat'], np.array([aug_data_cat])))
                        temp_data['y'] = np.vstack((temp_data['y'], np.array([aug_data_y])))
                    else:
                        temp_data['X_num'] = np.array([aug_data_num])
                        temp_data['X_cat'] = np.array([aug_data_cat])
                        temp_data['y'] = np.array([aug_data_y])
                
                if self.aug_data_all:
                    self.aug_data_all['X_num'] = np.vstack((self.aug_data_all['X_num'], temp_data['X_num']))
                    self.aug_data_all['X_cat'] = np.vstack((self.aug_data_all['X_cat'], temp_data['X_cat']))
                    self.aug_data_all['y'] = np.vstack((self.aug_data_all['y'], temp_data['y']))
                else:
                    self.aug_data_all['X_num'] = temp_data['X_num']
                    self.aug_data_all['X_cat'] = temp_data['X_cat']
                    self.aug_data_all['y'] = temp_data['y']
                
                aug_data_all_length = self.aug_data_all['y'].shape[0]

            if self.aug_data_all['X_cat'] is not None: 
                self.aug_data_all['X_cat'] = onehot_to_index(self.aug_data_all['X_cat'], self.cat_num_classes).astype(float)
            self.aug_data_all['y'] = onehot_to_index(self.aug_data_all['y'], [self.y_num_classes]).reshape(-1)

        return self.aug_data_all

    def change_aug(self, aug_data_num, aug_data_cat, aug_data_y, num_range, cat_range, y_range, aug_change_attr):
        attr = np.random.choice(np.arange(self.attr_num), size=aug_change_attr, replace = False)
        for i in attr: 
            if i < len(num_range):
                aug_data_num[i] = random.uniform(num_range[i][0], num_range[i][1])
            elif i < len(num_range) + len(cat_range) - 1: 
                current_cat = aug_data_cat[cat_range[i - len(num_range)]: cat_range[i + 1 - len(num_range)]]
                current_idx = np.argmax(current_cat)
                next_idx = random.choice([i for i in range(len(current_cat)) if i != current_idx]) 
                current_cat[current_idx] = 0
                current_cat[next_idx] = 1
                aug_data_cat[cat_range[i - len(num_range)]: cat_range[i + 1 - len(num_range)]] = current_cat
            else:
                if self.y_type == 'class': 
                    current_idx = np.argmax(aug_data_y)
                    next_idx = random.choice([i for i in range(len(aug_data_y)) if i != current_idx]) 
                    aug_data_y[current_idx] = 0
                    aug_data_y[next_idx] = 1
                else:
                    aug_data_num[i] = random.uniform(y_range[0], y_range[1]) 
        
        return aug_data_num, aug_data_cat, aug_data_y


    # def RFF_greedy_filter_process(self, save_path, threshold_param):
    #     random.seed(self.seed)
    #     record_df = pd.DataFrame(columns = ['block', 'start_MMD', 'end_MMD', 'start_data_num', 'end_data_num'])
    #     if self.y_type == 'class':
    #         with tqdm(total = len(self.block_idx)) as pbar:
    #             # iteration for each block
    #             for j in range(len(self.block_idx)):
    #                 block_filter_idx = np.ones(len(self.block_idx[j]))
    #                 MMD = np.linalg.norm(self.pretrain_RFF[j] - self.train_RFF, ord='fro')**2
    #                 count = 0
    #                 record = [j+1, deepcopy(MMD), 0, len(block_filter_idx), 0]

    #                 while sum(block_filter_idx) > 0:
    #                     best_improvement = 0.0
    #                     threshold = threshold_param * MMD / sum(block_filter_idx)
    #                     for i in np.where(block_filter_idx == 1)[0]:
    #                         temp_RFF = self.update_pretrain_RFF_delete(block_filter_idx, j, i)
    #                         temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='fro')**2
    #                         temp_improvement = MMD - temp_MMD
    #                         if temp_improvement >= best_improvement:
    #                             best_improvement = temp_improvement 
    #                             best_RFF = temp_RFF 
    #                             best_MMD = temp_MMD
    #                             best_i = i

    #                     if best_improvement > threshold:
    #                         block_filter_idx[best_i] = 0
    #                         self.filter_idx[self.block_idx[j][best_i]] = 0
    #                         self.pretrain_RFF[j] = best_RFF
    #                         MMD = best_MMD
    #                         count += 1
    #                     else: 
    #                         break

    #                 record[2] = MMD
    #                 record[4] = sum(block_filter_idx)
    #                 record_df.loc[j] = record

    #                 pbar.update(1)
    #                 pbar.set_description(f"Filter")
    #                 pbar.set_postfix(Filter_info = f'{count}/{len(block_filter_idx)}')

    #     elif self.y_type == 'regression':
    #         with tqdm(total = len(self.block_idx)) as pbar:
    #             for j in range(len(self.block_idx)):
    #                 block_filter_idx = np.ones(len(self.block_idx[j]))
    #                 MMD = np.linalg.norm(self.pretrain_RFF[j] - self.train_RFF, ord='2')**2
    #                 count = 0
    #                 record = [j+1, deepcopy(MMD), 0, len(block_filter_idx), 0]
    #                 while sum(block_filter_idx) > 0: # minimum sample number for accuracy 
    #                     best_improvement = 0.0
    #                     threshold = threshold_param * MMD / sum(block_filter_idx) # dynamic threshold for data filter
    #                     for i in np.where(block_filter_idx == 1)[0]:
    #                         temp_RFF = self.update_pretrain_RFF_delete(block_filter_idx, j, i)
    #                         temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='2')**2
    #                         temp_improvement = MMD - temp_MMD
    #                         if temp_improvement >= best_improvement:
    #                             best_improvement = temp_improvement 
    #                             best_RFF = temp_RFF 
    #                             best_MMD = temp_MMD
    #                             best_i = i

    #                     if best_improvement > threshold:
    #                         block_filter_idx[best_i] = 0
    #                         self.filter_idx[self.block_idx[j][best_i]] = 0
    #                         self.pretrain_RFF[j] = best_RFF
    #                         MMD = best_MMD
    #                         count += 1
    #                     else: 
    #                         break
                    
    #                 record[2] = MMD
    #                 record[4] = sum(block_filter_idx)
    #                 record_df.loc[j] = record

    #                 pbar.update(1)
    #                 pbar.set_description(f"Filter")
    #                 pbar.set_postfix(Filter_info = f'{count}/{len(block_filter_idx)}')
        
    #     record_df.loc[len(self.block_idx)] = ['all', '-', '-', len(self.filter_idx), sum(self.filter_idx)]
    #     record_df['start_data_num'] = record_df['start_data_num'].astype(int)
    #     record_df['end_data_num'] = record_df['end_data_num'].astype(int)
    #     record_df.to_csv(os.path.join(save_path, 'preprocess_record.csv'), index = False)

    #     print(f'Totally {len(self.filter_idx) - sum(self.filter_idx)} ({100 * (1 - sum(self.filter_idx)/len(self.filter_idx)) :.2f}%) data has been removed, {sum(self.filter_idx)} ({100 * sum(self.filter_idx)/len(self.filter_idx) :.2f}%) data remained')
    #     if sum(self.filter_idx) <= 0.05 * len(self.filter_idx):
    #         warnings.warn('Too much data has been removed. Please try larger threshold param or other pretrain dataset', UserWarning)
    #     print("-"*100)
    #     return self.filter_idx 


    # def RFF_random_filter_process(self, save_path, threshold_param, batch_size):
    #     random.seed(self.seed)
    #     record_df = pd.DataFrame(columns = ['block', 'start_MMD', 'end_MMD', 'start_data_num', 'end_data_num'])
    #     if self.y_type == 'class':
    #         with tqdm(total = len(self.block_idx)) as pbar:
    #             # iteration for each block
    #             for j in range(len(self.block_idx)):
    #                 block_filter_idx = np.ones(len(self.block_idx[j]))
    #                 MMD = np.linalg.norm(self.pretrain_RFF[j] - self.train_RFF, ord='fro')**2
    #                 count = 0
    #                 record = [j+1, deepcopy(MMD), 0, len(block_filter_idx), 0]
                    
    #                 while sum(block_filter_idx) > 0:
    #                     best_improvement = 0.0
    #                     threshold = threshold_param * MMD / sum(block_filter_idx)
    #                     bacth_list = generate_shuffle_block(np.where(block_filter_idx == 1)[0], batch_size)
    #                     stop = True
    #                     for batch in bacth_list:
    #                         for i in batch:
    #                             temp_RFF = self.update_pretrain_RFF_delete(block_filter_idx, j, i)
    #                             temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='fro')**2
    #                             temp_improvement = MMD - temp_MMD
    #                             if temp_improvement >= best_improvement:
    #                                 best_improvement = temp_improvement 
    #                                 best_RFF = temp_RFF 
    #                                 best_MMD = temp_MMD
    #                                 best_i = i
    #                         if best_improvement > threshold:
    #                             block_filter_idx[best_i] = 0
    #                             self.filter_idx[self.block_idx[j][best_i]] = 0
    #                             self.pretrain_RFF[j] = best_RFF
    #                             MMD = best_MMD
    #                             count += 1
    #                             stop = False
    #                             break
    #                     if stop: break

    #                 record[2] = MMD
    #                 record[4] = sum(block_filter_idx)
    #                 record_df.loc[j] = record

    #                 pbar.update(1)
    #                 pbar.set_description(f"Filter")
    #                 pbar.set_postfix(Filter_info = f'{count}/{len(block_filter_idx)}')

    #     elif self.y_type == 'regression':
    #         with tqdm(total = len(self.block_idx)) as pbar:
    #             for j in range(len(self.block_idx)):
    #                 block_filter_idx = np.ones(len(self.block_idx[j]))
    #                 MMD = np.linalg.norm(self.pretrain_RFF[j] - self.train_RFF, ord='2')**2
    #                 count = 0
    #                 record = [j+1, deepcopy(MMD), 0, len(block_filter_idx), 0]
    #                 while sum(block_filter_idx) > 0: # minimum sample number for accuracy 
    #                     best_improvement = 0.0
    #                     threshold = threshold_param * MMD / sum(block_filter_idx) # dynamic threshold for data filter
    #                     for i in np.where(block_filter_idx == 1)[0]:
    #                         temp_RFF = self.update_pretrain_RFF_delete(block_filter_idx, j, i)
    #                         temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='2')**2
    #                         temp_improvement = MMD - temp_MMD
    #                         if temp_improvement >= best_improvement:
    #                             best_improvement = temp_improvement 
    #                             best_RFF = temp_RFF 
    #                             best_MMD = temp_MMD
    #                             best_i = i

    #                     if best_improvement > threshold:
    #                         block_filter_idx[best_i] = 0
    #                         self.filter_idx[self.block_idx[j][best_i]] = 0
    #                         self.pretrain_RFF[j] = best_RFF
    #                         MMD = best_MMD
    #                         count += 1
    #                     else: 
    #                         break
                    
    #                 record[2] = MMD
    #                 record[4] = sum(block_filter_idx)
    #                 record_df.loc[j] = record

    #                 pbar.update(1)
    #                 pbar.set_description(f"Filter")
    #                 pbar.set_postfix(Filter_info = f'{count}/{len(block_filter_idx)}')
        
    #     record_df.loc[len(self.block_idx)] = ['all', '-', '-', len(self.filter_idx), sum(self.filter_idx)]
    #     record_df['start_data_num'] = record_df['start_data_num'].astype(int)
    #     record_df['end_data_num'] = record_df['end_data_num'].astype(int)
    #     record_df.to_csv(os.path.join(save_path, 'preprocess_record.csv'), index = False)

    #     print(f'Totally {len(self.filter_idx) - sum(self.filter_idx)} ({100 * (1 - sum(self.filter_idx)/len(self.filter_idx)) :.2f}%) data has been removed, {sum(self.filter_idx)} ({100 * sum(self.filter_idx)/len(self.filter_idx) :.2f}%) data remained')
    #     if sum(self.filter_idx) <= 0.05 * len(self.filter_idx):
    #         warnings.warn('Too much data has been removed. Please try larger threshold param or other pretrain dataset', UserWarning)
    #     print("-"*100)
    #     return self.filter_idx


def decide_imputation(D: Dataset):
    num_miss = False
    cat_miss = False
    if D.X_num is not None:
        num_miss = np.isnan(D.X_num['pretrain']).any()
    if D.X_cat is not None:
        cat_miss = np.isnan(D.X_cat['pretrain']).any()
    return int(num_miss | cat_miss)








'''

def pretrain_data_imputation(D: Dataset, rho, seed = 0):
    random.seed(seed)
    num_miss = []
    cat_miss = []
    
    if D.X_num is not None:
        for i in range(D.X_num['pretrain'].shape[1]):
            if np.isnan(D.X_num['pretrain'][:, i]).any():
                num_miss.append(i)
    if D.X_cat is not None:
        for i in range(D.X_cat['pretrain'].shape[1]): 
            if np.isnan(D.X_cat['pretrain'][:, i]).any():
                cat_miss.append(i)
    
    miss_attr = len(num_miss) + len(cat_miss)
    if miss_attr == 0:
        print('No missing data, need to change config')
        return D
    
    rho_attr = rho/miss_attr 

    if len(num_miss) > 0:
        for i in num_miss:
            element, count = np.unique(D.X_num['train'][:, i], return_counts = True)
            count = count + np.sqrt(2.0) * np.sqrt(1 / (2*rho_attr)) * np.random.randn(len(count))
            count = np.where(count < 0, 0.0, count)
            miss_idx = np.isnan(D.X_num['pretrain'][:, i])

            D.X_num['pretrain'][miss_idx, i] = np.random.choice(
                element,
                size = sum(miss_idx),
                p = count/np.sum(count)
            )
    if len(cat_miss) > 0:
        for i in cat_miss:
            element, count = np.unique(D.X_cat['train'][:, i], return_counts = True)
            count = count + np.sqrt(2.0) * np.sqrt(1 / (2*rho_attr)) * np.random.randn(len(count))
            count = np.where(count < 0, 0.0, count)
            miss_idx = np.isnan(D.X_cat['pretrain'][:, i])

            D.X_cat['pretrain'][miss_idx, i] = np.random.choice(
                element,
                size = sum(miss_idx),
                p = count/np.sum(count)
            )
    print('Finish pretrain data imputation')
    return D



def prepare_datapreprocess_dataloader(dataset, split, batch_size, num_size = 0, cat_size = [0], y_size = 1, return_dataloader = True):
    num_part = None
    cat_part = None
    if num_size != 0:
        num_part = torch.tensor(dataset.X_num[split], dtype = torch.float32)
    if np.sum(cat_size) != 0:
        cat_part = torch.tensor(dataset.X_cat[split], dtype = torch.int64)
    if y_size == 1:
        y_part = torch.tensor(dataset.y[split], dtype = torch.float32).reshape(-1,1)
    else:
        y_part = torch.tensor(dataset.y[split], dtype = torch.int64).reshape(-1,1)
    
    if return_dataloader == False:
        return num_part, cat_part, y_part
    else:
        if num_size == 0:
            ds = TensorDataset(cat_part, y_part)
        elif np.sum(cat_size) == 0:
            ds = TensorDataset(num_part, y_part)
        else:
            ds = TensorDataset(num_part, cat_part, y_part)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)


def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        assert policy is None
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        raise ValueError('numerical NaN policy:', policy)
    return dataset

def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            raise ValueError('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new

class RandomFourierFeatureProcesser:
    def __init__(self, dataset: Dataset, feature_dim, dp_epsilon = None, dp_delta = None, block=1024, seed=0):
        self.feature_dim = feature_dim 
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta 
        self.seed = seed

        self.pretrain_m = dataset.y['pretrain'].shape[0]
        self.train_m = dataset.y['train'].shape[0]
        self.num_num_classes = dataset.X_num['train'].shape[1]   # a number of columns of x_num
        self.cat_num_classes = np.array(dataset.get_category_sizes('train'))   # a list of category number of x_cat
        self.y_num_classes = dataset.y_info['num_classes']   # category number of y

        if self.num_num_classes is None:
            self.num_num_classes = 0
        if self.cat_num_classes is None:
            self.cat_num_classes = np.array([0])
        if self.y_num_classes == None:
            self.y_type = 'regression'
        else: 
            self.y_type = 'class'
        
        for part in ['pretrain', 'train']:
            if dataset.X_cat is not None:
                dataset.X_cat[part] = index_to_onehot(dataset.X_cat[part], self.cat_num_classes)
            if self.y_type == 'class':
                dataset.y[part] = index_to_onehot(dataset.y[part], np.array([self.y_num_classes]))
            elif self.y_type == 'regression':
                dataset.X_num[part] = np.hstack(dataset.y[part], dataset.X_num[part]) 
        
        print("-"*100)
        self.dataset = dataset
        self.filter_idx = np.ones(len(dataset.y['pretrain']))
        self.block_idx = generate_shuffle_block(len(dataset.y['pretrain']), 1024)


    def RFF(self, x:np.array, feature_type):
        random.seed(self.seed)
        if feature_type == 'num':
            w = np.random.randn(self.feature_dim // 2, self.num_num_classes)
            w = np.vstack((w, w.copy()))

            phi = np.matmul(x, w.T)
            cos_half = np.cos(phi[: ,: self.feature_dim // 2])
            sin_half = np.sin(phi[: ,self.feature_dim // 2 :])
            phi[:, : self.feature_dim // 2] = cos_half
            phi[:, self.feature_dim // 2 :] = sin_half
        
            return np.sqrt(2/self.feature_dim) * phi
        
        if feature_type == 'cat':
            return x / np.sqrt(np.sum(self.cat_num_classes))

        if feature_type == 'y': 
            return x / np.sqrt(self.y_num_classes)

        else: 
            raise ValueError('Invalid kernel type')
        
    
    def initialize_fourier_feature(self):
        random.seed(self.seed)
        if self.y_type == 'class':
            # RFF of pretrain data
            fx = np.hstack((self.RFF(self.dataset.X_num['pretrain'], 'num'), self.RFF(self.dataset.X_cat['pretrain'], 'cat')))
            fy = self.RFF(self.dataset.y['pretrain'], 'y')

            self.pretrain_RFF_list = np.zeros((self.pretrain_m, self.feature_dim + np.sum(self.cat_num_classes), self.y_num_classes))
            for i in range(self.pretrain_m):
                self.pretrain_RFF_list[i] = np.outer(fx[i], fy[i])
            self.pretrain_RFF = np.mean(self.pretrain_RFF_list, axis=0)


            # differential private RFF of pretrain data
            fx = np.hstack((self.RFF(self.dataset.X_num['train'], 'num'), self.RFF(self.dataset.X_cat['train'], 'cat')))
            fy = self.RFF(self.dataset.y['train'], 'y')

            self.train_RFF = np.zeros((self.feature_dim + np.sum(self.cat_num_classes), self.y_num_classes))
            for i in range(self.train_m):
                self.train_RFF += np.outer(fx[i], fy[i]) / self.train_m 

            sigma = np.sqrt(2*np.log(1.25/self.dp_delta))/self.dp_epsilon
            RFF_noise = 2/self.train_m * sigma * np.random.randn(self.feature_dim + np.sum(self.cat_num_classes), self.y_num_classes)
            self.train_RFF += RFF_noise

        
        if self.y_type == 'regression':
            # RFF of pretrain data
            fx = np.hstack((self.RFF(self.dataset.X_num['pretrain'], 'num'), self.RFF(self.dataset.X_cat['pretrain'], 'cat')))
            self.pretrain_RFF_list = np.zeros(self.pretrain_m, self.feature_dim + np.sum(self.cat_num_classes) + 1)
            for i in range(self.pretrain_m):
                self.pretrain_RFF_list[i] = fx[i]
            self.pretrain_RFF = np.mean(fx, axis=0)

            # differential private RFF of pretrain data
            fx = np.hstack((self.RFF(self.dataset.X_num['train'], 'num'), self.RFF(self.dataset.X_cat['train'], 'cat')))
            self.train_RFF = np.mean(fx, axis=0)

            sigma = np.sqrt(2*np.log(1.25/self.dp_delta))/self.dp_epsilon
            RFF_noise = 2/self.train_m * sigma * np.random.randn(self.feature_dim + np.sum(self.cat_num_classes) + 1)
            self.train_RFF += RFF_noise 
        
        print("finish initializing RFF")
    
    def update_pretrain_RFF(self, i):
        RFF = sum(self.filter_idx) * self.pretrain_RFF - self.pretrain_RFF_list[i] 
        RFF = RFF/(sum(self.filter_idx) - 1) 

        return RFF

    def RFF_process(self):

        random.seed(self.seed)
        if self.y_type == 'class':
            self.MMD = np.linalg.norm(self.pretrain_RFF - self.train_RFF, ord='fro')**2
            count = 0
            fail_count = 0
            fail_count_bound = sum(self.filter_idx) 
            threshold = 0.1 * self.MMD/sum(self.filter_idx)
            while sum(self.filter_idx) > 0:
                i = np.random.choice(np.where(self.filter_idx == 1)[0])
                temp_RFF = self.update_pretrain_RFF(i)
                temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='fro')**2
                temp_improvement = self.MMD - temp_MMD

                if temp_improvement > threshold:
                    self.filter_idx[i] = 0
                    self.pretrain_RFF = temp_RFF
                    self.MMD = temp_MMD
                    fail_count_bound = sum(self.filter_idx)
                    print(f'Number of removed data: {count:4d}', end='\r')
                    threshold = 0.1 * self.MMD/sum(self.filter_idx)
                    count += 1
                    fail_count = 0
                else: 
                    fail_count += 1
                    if fail_count >= fail_count_bound:
                        break

        elif self.y_type == 'regression':
            self.MMD = np.linalg.norm(self.pretrain_RFF - self.train_RFF, ord='2')**2
            count = 0
            fail_count = 0
            fail_count_bound = sum(self.filter_idx) 
            threshold = 0.1 * self.MMD/sum(self.filter_idx)
            while sum(self.filter_idx) > 0:
                i = np.random.choice(np.where(self.filter_idx == 1)[0])
                temp_RFF = self.update_pretrain_RFF(i)
                temp_MMD = np.linalg.norm(temp_RFF - self.train_RFF, ord='2')**2
                temp_improvement = self.MMD - temp_MMD

                if temp_improvement > threshold:
                    self.filter_idx[i] = 0
                    self.pretrain_RFF = temp_RFF
                    self.MMD = temp_MMD
                    fail_count_bound = sum(self.filter_idx)
                    print(f'Number of removed data: {count:4d}', end='\r')
                    threshold = 0.1 * self.MMD/sum(self.filter_idx)
                    count += 1
                    fail_count = 0
                else: 
                    fail_count += 1
                    if fail_count >= fail_count_bound:
                        break
        
        print(f'{len(self.filter_idx) - sum(self.filter_idx)} data has been removed, {sum(self.filter_idx)} data remained')
        print("-"*100)
        return self.filter_idx

'''