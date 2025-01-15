import numpy as np
import pandas as pd
import os
import json
import sklearn
from functools import reduce
from preprocess_common.preprocess import * 

import sklearn.preprocessing
from method.AIM.mbi.Domain import Domain

class Dataset:
    def __init__(
        self, 
        df, 
        domain, 
        weights=None, 
    ):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(
            df.columns
        ), "data must contain domain attributes"
        assert weights is None or df.shape[0] == weights.size
        self.domain = domain
        self.df = df.loc[:, domain.attrs]
        self.weights = weights

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object 
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load_old(path):
        """ Load data into a dataset object

        :param path: path to three/two npy file, and the domain file
        :(deleted) param domain: path to json file encoding the domain information
        """
        # df = pd.read_csv(path)
        X_num = None 
        X_cat = None 
        num_col = 0
        cat_col = 0
        cat_encoder = None

        if os.path.exists(os.path.join(path, 'X_num_train.npy')):
            X_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
            num_col = X_num.shape[1]
            n_bins = [min(len(set(X_num[:,i])), 500) for i in range(X_num.shape[1])]
            num_encoder = sklearn.preprocessing.KBinsDiscretizer(
                n_bins= n_bins,
                encode='ordinal', 
                strategy='uniform'
            )
            X_num = num_encoder.fit_transform(X_num) 

        if os.path.exists(os.path.join(path, 'X_cat_train.npy')):
            X_cat = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)
            cat_col = X_cat.shape[1]
            cat_encoder = sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            X_cat = cat_encoder.fit_transform(X_cat)

        y = np.load(os.path.join(path, 'y_train.npy'), allow_pickle=True)

        col_name = [f'num_attr_{i}' for i in range(1, num_col + 1)] + [f'cat_attr_{i}' for i in range(1, cat_col + 1)] + ['y_attr']

        if X_num is None:
            df = pd.DataFrame(np.concatenate((X_cat, y.reshape(-1,1)), axis=1), columns=col_name)
        elif X_cat is None:
            df = pd.DataFrame(np.concatenate((X_num, y.reshape(-1,1)), axis=1), columns=col_name)
        else:
            df = pd.DataFrame(np.concatenate((X_num, X_cat, y.reshape(-1,1)), axis=1), columns=col_name)

        config = json.load(open(os.path.join(path, 'domain.json')))
        for i in range(1, num_col + 1):
            config[f'num_attr_{i}'] = min(config[f'num_attr_{i}'], len(set(X_num[:, i-1])))

        domain = Domain(config.keys(), config.values())
        
        return Dataset(df, domain, num_encoder, cat_encoder, num_col = num_col, cat_col = cat_col) 
    
    @staticmethod
    def load_unused(path, rho, num_prep, rare_threshold):
        """ Load data into a dataset object

        :param path: path to three/two npy file, and the domain file
        :(deleted) param domain: path to json file encoding the domain information
        """
        # df = pd.read_csv(path)
        X_num = None 
        X_cat = None 
        num_encoder = None
        cat_encoder = None
        num_col = 0
        cat_col = 0

        if os.path.exists(os.path.join(path, 'X_num_train.npy')):
            X_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
            num_col = X_num.shape[1]
        if os.path.exists(os.path.join(path, 'X_cat_train.npy')):
            X_cat = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)
            cat_col = X_cat.shape[1]
        y = np.load(os.path.join(path, 'y_train.npy'), allow_pickle=True)
        num_rho, cat_rho = calculate_rho_allocate(X_num, X_cat, num_prep)

        if X_num is not None:
            num_encoder = discretizer(num_prep, num_rho * 0.1 * rho)
            X_num = num_encoder.fit_transform(X_num)
        if X_cat is not None:
            cat_encoder = rare_merger(cat_rho * 0.1 * rho, rare_threshold=rare_threshold)
            X_cat = cat_encoder.fit_transform(X_cat)

        col_name = [f'num_attr_{i}' for i in range(1, num_col + 1)] + [f'cat_attr_{i}' for i in range(1, cat_col + 1)] + ['y_attr']

        if X_num is None:
            df = pd.DataFrame(np.concatenate((X_cat, y.reshape(-1,1)), axis=1), columns=col_name)
        elif X_cat is None:
            df = pd.DataFrame(np.concatenate((X_num, y.reshape(-1,1)), axis=1), columns=col_name)
        else:
            df = pd.DataFrame(np.concatenate((X_num, X_cat, y.reshape(-1,1)), axis=1), columns=col_name)
        

        config = json.load(open(os.path.join(path, 'domain.json')))
        for i in range(1, num_col + 1):
            config[f'num_attr_{i}'] = min(config[f'num_attr_{i}'], len(set(X_num[:, i-1])))
        for i in range(1, cat_col + 1):
            config[f'cat_attr_{i}'] = min(config[f'cat_attr_{i}'], len(set(X_cat[:, i-1])))

        domain = Domain(config.keys(), config.values())
        
        return Dataset(df, domain), num_rho, cat_rho
    
    @staticmethod
    def load_from_dataset(X_num, X_cat, y): 
        num_col = 0
        cat_col = 0

        if X_num is None:
            cat_col = X_cat.shape[1]
            col_name = [f'cat_attr_{i}' for i in range(1, cat_col + 1)] + ['y_attr']
            df = pd.DataFrame(np.concatenate((X_cat, y.reshape(-1,1)), axis=1), columns=col_name)
        elif X_cat is None:
            num_col = X_num.shape[1]
            col_name = [f'num_attr_{i}' for i in range(1, num_col + 1)] + ['y_attr']
            df = pd.DataFrame(np.concatenate((X_num, y.reshape(-1,1)), axis=1), columns=col_name)
        else:
            num_col = X_num.shape[1]
            cat_col = X_cat.shape[1]
            col_name = [f'num_attr_{i}' for i in range(1, num_col + 1)] + [f'cat_attr_{i}' for i in range(1, cat_col + 1)] + ['y_attr']
            df = pd.DataFrame(np.concatenate((X_num, X_cat, y.reshape(-1,1)), axis=1), columns=col_name)
        
        domain = Domain(col_name, df.nunique().tolist())
        return Dataset(df, domain, None) 


    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        return self.df.shape[0]

    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n + 1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans

    def save_data_npy(self, path, preprocesser=None): 
        if preprocesser is not None:
            assert ((preprocesser.num_col is not None) and (preprocesser.cat_col is not None))

        preprocesser.reverse_data(self.df, path)

        # if self.num_col > 0:
        #     x_num = self.df.iloc[:, 0:self.num_col].to_numpy()
        #     if post_processer.num_encoder is not None:
        #         x_num = post_processer.num_encoder.inverse_transform(x_num).astype(float)
        #     np.save(os.path.join(path, 'X_num_train.npy'), x_num)

        # if self.cat_col > 0:
        #     x_cat = self.df.iloc[:, self.num_col:self.num_col+self.cat_col].to_numpy()
        #     if post_processer.cat_encoder is not None:
        #         x_cat = post_processer.cat_encoder.inverse_transform(x_cat).astype(str)
        #     np.save(os.path.join(path, 'X_cat_train.npy'), x_cat) 
        
        # y = self.df.iloc[:, -1].to_numpy().reshape(-1).astype(int)
        # np.save(os.path.join(path, 'y_train.npy'), y)