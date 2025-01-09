import numpy as np 
import pandas as pd
import os
import json
import math
import sklearn.preprocessing
from preprocess_common.preprocess import * 
from util.rho_cdp import cdp_rho

class data_preporcesser_common():
    def __init__(self, args):
        self.num_encoder = None
        self.cat_encoder = None
        self.num_col = 0
        self.cat_col = 0
        self.args = args
        self.cat_output_type = 'one_hot' if self.args.method == 'merf' else 'ordinal'

    def load_data(self, path, rho):
        # preprocesser and column info will be saved in this class
        # dataframe, domain domain, portion of rho will be returned

        num_prep = self.args.num_preprocess
        rare_threshold = self.args.rare_threshold
        print(f'Numerical discretizer is {num_prep}')

        X_num = None 
        X_cat = None 

        if os.path.exists(os.path.join(path, 'X_num_train.npy')):
            X_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'X_cat_train.npy')):
            X_cat = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)
        y = np.load(os.path.join(path, 'y_train.npy'), allow_pickle=True)
        num_divide, cat_divide = calculate_rho_allocate(X_num, X_cat, num_prep)

        if (rho == 0) and (num_divide + cat_divide > 0):
            if self.args.epsilon > 0:
                rho = cdp_rho(0.1*(num_divide+cat_divide)*self.args.epsilon, 0.1*(num_divide+cat_divide)*self.args.delta)
            else:
                rho = cdp_rho(0.1*(num_divide+cat_divide)*1.0, 0.1*(num_divide+cat_divide)*1e-5) # this is the default value setting
    
        if X_num is not None:
            if num_prep != 'none':
                ord = False if self.args.method in ['rap', 'gsd'] else True
                self.num_encoder = discretizer(num_prep, num_divide * 0.1 * rho, ord=ord)
            else:
                self.num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
            X_num = self.num_encoder.fit_transform(X_num)
            self.num_col = X_num.shape[1]
        if X_cat is not None:
            self.cat_encoder = rare_merger(cat_divide * 0.1 * rho, rare_threshold=rare_threshold, output_type=self.cat_output_type)
            X_cat = self.cat_encoder.fit_transform(X_cat)
            self.cat_col = X_cat.shape[1]

        if self.args.method in ('merf', 'ddpm'):
            df = {
                "X_num": X_num,
                "X_cat": X_cat,
                "y": y
            }
            domain = {
                "X_num": [len(set(X_num[:, i])) for i in range(self.num_col)] if X_num is not None else [],
                "X_cat": [len(cat) for cat in self.cat_encoder.ordinal_encoder.categories_] if X_cat is not None else [],
                "y": [len(set(y))]
            }
        else:
            col_name = [f'num_attr_{i}' for i in range(1, self.num_col + 1)] + [f'cat_attr_{i}' for i in range(1, self.cat_col + 1)] + ['y_attr']

            if X_num is None:
                df = pd.DataFrame(np.concatenate((X_cat, y.reshape(-1,1)), axis=1), columns=col_name)
            elif X_cat is None:
                df = pd.DataFrame(np.concatenate((X_num, y.reshape(-1,1)), axis=1), columns=col_name)
            else:
                df = pd.DataFrame(np.concatenate((X_num, X_cat, y.reshape(-1,1)), axis=1), columns=col_name)

            domain = json.load(open(os.path.join(path, 'domain.json')))
            for i in range(1, self.num_col + 1):
                domain[f'num_attr_{i}'] = min(domain[f'num_attr_{i}'], len(set(X_num[:, i-1])))
            for i in range(1, self.cat_col + 1):
                domain[f'cat_attr_{i}'] = min(domain[f'cat_attr_{i}'], len(set(X_cat[:, i-1]))) 
    
        return df, domain, 0.1*(num_divide + cat_divide)


    def reverse_data(self, df, path=None):
        if isinstance(df, pd.DataFrame):
            df = df.to_numpy()
        
        x_num = None 
        x_cat = None 
        if self.num_col > 0:
            x_num = df[:, 0:self.num_col]
            if x_num.ndim == 1:
                x_num = x_num.reshape(-1,1)
            
            if self.num_encoder is not None:
                x_num = self.num_encoder.inverse_transform(x_num).astype(float)
            if path is not None:
                np.save(os.path.join(path, 'X_num_train.npy'), x_num)

        if self.cat_col > 0:
            x_cat = df[:, self.num_col:self.num_col+self.cat_col]
            if x_cat.ndim == 1:
                x_cat = x_cat.reshape(-1,1)
            
            if self.cat_encoder is not None:
                x_cat = self.cat_encoder.inverse_transform(x_cat).astype(str)
            if path is not None:
                np.save(os.path.join(path, 'X_cat_train.npy'), x_cat) 
        
        y = df[:, -1].reshape(-1).astype(int)
        if path is not None:
            np.save(os.path.join(path, 'y_train.npy'), y) 
        
        return x_num, x_cat, y