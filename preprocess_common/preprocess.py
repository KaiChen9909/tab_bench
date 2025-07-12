###################################################################

# This file contain some commonly used preprocess (discretize) method 

###################################################################

import numpy as np 
import pandas as pd
import sklearn
import sklearn.preprocessing
from copy import deepcopy
from preprocess_common.privtree import privtree
from preprocess_common.dawa import dawa

def laplace_noise(Lambda):
    return np.random.laplace(loc=0, scale=Lambda)

class discretizer():
    def __init__(self, bins_method, rho, bin_number=100, ord = True):
        self.bins_method = bins_method
        self.rho = rho
        self.bin_number = bin_number
        self.ord = ord

    def fit(self, data):
        if self.bins_method == 'none':
            return None

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.n_bins = [min(len(set(data[:,i])), self.bin_number) for i in range(data.shape[1])]
        self.columns_for_kbins = [i for i in range(len(self.n_bins)) if self.n_bins[i] == self.bin_number]
        self.columns_for_ordinal = [i for i in range(len(self.n_bins)) if self.n_bins[i] < self.bin_number]

        if len(self.columns_for_kbins) > 0:
            if self.bins_method == 'uniform_kbins':
                self.kbin_encoder = sklearn.preprocessing.KBinsDiscretizer(
                    n_bins= self.bin_number,
                    encode='ordinal', 
                    strategy='uniform'
                )
            elif self.bins_method == 'exp_kbins':
                self.min_value = np.array([min(data[:,i]) for i in self.columns_for_kbins]) - 1e-2
                self.kbin_encoder = sklearn.preprocessing.KBinsDiscretizer(
                    n_bins= self.bin_number,
                    encode='ordinal', 
                    strategy='uniform'
                )
            elif self.bins_method == 'privtree':
                self.kbin_encoder = privtree(
                    self.rho
                )
            elif self.bins_method == 'dawa':
                self.kbin_encoder = dawa(
                    self.rho
                )
            
            if self.bins_method == 'exp_kbins':
                self.kbin_encoder.fit(np.log2(data[:, self.columns_for_kbins] - self.min_value))
            else: 
                self.kbin_encoder.fit(data[:, self.columns_for_kbins])
        else:
            print('No need for binning')

        
        return None


    def transform(self, data):
        if self.bins_method == 'none':
            return data

        encoded_data = data.copy()
        if len(self.columns_for_kbins) > 0:
            if self.bins_method == 'exp_kbins':
                encoded_data[:, self.columns_for_kbins] = self.kbin_encoder.transform(np.log2(data[:, self.columns_for_kbins]  - self.min_value))
            else:
                encoded_data[:, self.columns_for_kbins] = self.kbin_encoder.transform(data[:, self.columns_for_kbins])
        
        if self.ord:
            self.ord_encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='error'
            )
            encoded_data = self.ord_encoder.transform(encoded_data)
        else:
            self.ord_encoder = None

        return encoded_data


    def fit_transform(self, data):
        if self.bins_method == 'none':
            return data
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        encoded_data = data.copy()
        self.n_bins = [min(len(set(data[:,i])), self.bin_number) for i in range(data.shape[1])]
        self.columns_for_kbins = [i for i in range(len(self.n_bins)) if self.n_bins[i] == self.bin_number]
        self.columns_for_ordinal = [i for i in range(len(self.n_bins)) if self.n_bins[i] < self.bin_number]

        if len(self.columns_for_kbins) > 0:
            if self.bins_method == 'uniform_kbins':
                self.kbin_encoder = sklearn.preprocessing.KBinsDiscretizer(
                    n_bins= self.bin_number,
                    encode='ordinal', 
                    strategy='uniform'
                )
                encoded_data[:, self.columns_for_kbins] = self.kbin_encoder.fit_transform(data[:, self.columns_for_kbins])
            elif self.bins_method == 'exp_kbins':
                self.min_value = np.array([min(data[:,i]) for i in self.columns_for_kbins]) - 1e-2
                self.kbin_encoder = sklearn.preprocessing.KBinsDiscretizer(
                    n_bins= self.bin_number,
                    encode='ordinal', 
                    strategy='uniform'
                )
                encoded_data[:, self.columns_for_kbins] = self.kbin_encoder.fit_transform(np.log2(data[:, self.columns_for_kbins]  - self.min_value))
            elif self.bins_method == 'privtree':
                self.kbin_encoder = privtree(
                    self.rho
                )
                encoded_data[:, self.columns_for_kbins] = self.kbin_encoder.fit_transform(data[:, self.columns_for_kbins])
            elif self.bins_method == 'dawa':
                self.kbin_encoder = dawa(
                    self.rho
                )
                encoded_data[:, self.columns_for_kbins] = self.kbin_encoder.fit_transform(data[:, self.columns_for_kbins])
        else:
            print('No need for binning')

        if self.ord:
            self.ord_encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='error'
            )
            encoded_data = self.ord_encoder.fit_transform(encoded_data)
        else:
            self.ord_encoder = None

        return encoded_data 
    

    def inverse_transform(self, data):
        if self.bins_method == 'none':
            return data 
        
        decoded_data = data.copy()
        if self.ord_encoder is not None:
            decoded_data = self.ord_encoder.inverse_transform(data)

        if len(self.columns_for_kbins) > 0:
            if self.bins_method == 'exp_kbins':
                decoded_data[:, self.columns_for_kbins] = np.exp2(self.inverse_bin(decoded_data[:, self.columns_for_kbins]))
            elif self.bins_method == 'uniform_kbins':
                decoded_data[:, self.columns_for_kbins] = self.inverse_bin(decoded_data[:, self.columns_for_kbins])
            else:
                decoded_data[:, self.columns_for_kbins] = self.kbin_encoder.inverse_transform(decoded_data[:, self.columns_for_kbins])

        return decoded_data
    

    def inverse_bin(self, binned_data):
        binned = binned_data.astype(int)
        edges = self.kbin_encoder.bin_edges_
        sampled = np.empty_like(binned, dtype=float)

        for j in range(binned.shape[1]):
            left = edges[j][binned[:, j]]
            right = edges[j][binned[:, j] + 1]
            sampled[:, j] = np.random.uniform(left, right)
        return sampled


# def single_uniform_kbin(data):
#     n_bins = min(len(set(data)), self.bin_number)
#     if n_bins == self.bin_number:
#         num_encoder = sklearn.preprocessing.KBinsDiscretizer(
#             n_bins= n_bins,
#             encode='ordinal', 
#             strategy='uniform'
#         )
#         data = num_encoder.fit_transform(data)
#         return data, num_encoder 
#     else:
#         num_encoder = sklearn.preprocessing.OrdinalEncoder(
#             handle_unknown='use_encoded_value',
#             unknown_value=0
#         )
#         data = num_encoder.fit_transform(data)
#         return data, num_encoder 


class rare_merger():
    def __init__(self, rho, output_type = 'ordinal', rare_threshold=0.002, unique_threshold=100, default_rare_encode_value = 'R'):
        self.rho = rho
        self.output_type = output_type
        self.unique_threshold = unique_threshold
        self.rare_threshold = rare_threshold
        self.default_rare_encode_value = default_rare_encode_value
    

    def fit(self, data):
        assert (
            isinstance(data, np.ndarray)
        ), 'Must input an array data'
        assert (
            np.issubdtype(data.dtype, np.str_) or np.issubdtype(data.dtype, np.bytes_) or data.dtype == np.dtype('O')
        ), 'Categorical data is expected to be string or object type'
        assert(
            ~ np.any(data == self.default_rare_encode_value)
        ), 'Please change a rare encode value'

        encoded_data = deepcopy(data)
        self.rare_values = []
        self.columns_for_merge = []

        if self.rare_threshold >= 0:
            self.unique_count = [len(set(data[:,i])) for i in range(data.shape[1])]
            self.columns_for_merge = [i for i in range(len(self.unique_count)) if self.unique_count[i] >= self.unique_threshold]
            if len(self.columns_for_merge) > 0:
                K = len(self.columns_for_merge)
                sigma = np.sqrt(K/(2*self.rho))

                for i in self.columns_for_merge:
                    rare_value = []
                    x = encoded_data[:, i]
                    x_dict = self.noisy_count(x, self.rho/K)
                    n = sum(x_dict.values())
                    for k,v in x_dict.items():
                        if v <= n * max(self.rare_threshold, 3*sigma/n):
                            rare_value.append(k)
                    self.rare_values.append(rare_value)
            else:
                print(f'No need for merge under threshold {self.rare_threshold}')
        else:
            print(f'No need for merge under threshold {self.rare_threshold}')
        
        if self.output_type == 'ordinal':
            self.ordinal_encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=np.nan,
            )
        else:
            self.ordinal_encoder = sklearn.preprocessing.OneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore'
            )
        self.ordinal_encoder.fit(encoded_data)
            
    
    def transform(self, data):
        assert (
            isinstance(data, np.ndarray)
        ), 'Must input an array data'
        assert (
            np.issubdtype(data.dtype, np.str_) or np.issubdtype(data.dtype, np.bytes_) or data.dtype == np.dtype('O')
        ), 'Categorical data is expected to be string or object type'

        encoded_data = deepcopy(data)
        if len(self.columns_for_merge) > 0:
            for i in self.columns_for_merge:
                rare_value = []
                x = encoded_data[:, i]
                rare_value = self.rare_values[i]
                for k in rare_value:
                    x[x == k] = self.default_rare_encode_value
                # encoded_data[:, i] = x
        else:
            print('No need for merge')
        
        encoded_data = self.ordinal_encoder.transform(encoded_data)
        np.nan_to_num(encoded_data, nan=0)
        
        return encoded_data


    def fit_transform(self, data):
        assert (
            isinstance(data, np.ndarray)
        ), 'Must input an array data'
        assert (
            np.issubdtype(data.dtype, np.str_) or np.issubdtype(data.dtype, np.bytes_) or data.dtype == np.dtype('O')
        ), 'Categorical data is expected to be string or object type'
        assert(
            ~ np.any(data == self.default_rare_encode_value)
        ), 'Please change a rare encode value'

        encoded_data = deepcopy(data)
        self.rare_values = []
        self.columns_for_merge = []

        if self.rare_threshold >= 0:
            self.unique_count = [len(set(data[:,i])) for i in range(data.shape[1])]
            self.columns_for_merge = [i for i in range(len(self.unique_count)) if self.unique_count[i] >= self.unique_threshold]
            if len(self.columns_for_merge) > 0:
                K = len(self.columns_for_merge)
                sigma = np.sqrt(K/(2*self.rho))

                for i in self.columns_for_merge:
                    rare_value = []
                    x = encoded_data[:, i]
                    x_dict = self.noisy_count(x, self.rho/K)
                    n = sum(x_dict.values())

                    for k,v in x_dict.items():
                        if v <= n * max(self.rare_threshold, 3*sigma/n):
                            rare_value.append(k)
                            x[x == k] = self.default_rare_encode_value
                    self.rare_values.append(rare_value)
                    # encoded_data[:, i] = x
            else:
                print(f'No need for merge under threshold {self.rare_threshold}')
        else:
            print(f'No need for merge under threshold {self.rare_threshold}')
        
        if self.output_type == 'ordinal':
            self.ordinal_encoder = sklearn.preprocessing.OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=np.nan,
            )
        else:
            self.ordinal_encoder = sklearn.preprocessing.OneHotEncoder(
                sparse_output=False, 
                handle_unknown='ignore'
            )
        encoded_data = self.ordinal_encoder.fit_transform(encoded_data)
        np.nan_to_num(encoded_data, nan=0)

        return encoded_data


    def inverse_transform(self, data):
        assert (
            isinstance(data, np.ndarray)
        ), 'Must input an array data'

        decoded_data = self.ordinal_encoder.inverse_transform(data)

        if len(self.columns_for_merge) > 0:
            for i in range(len(self.columns_for_merge)):
                x = decoded_data[:, self.columns_for_merge[i]]
                id = (x == self.default_rare_encode_value)
                x[id] = np.random.choice(self.rare_values[i], size=sum(id), replace=True)
        else:
            print('No need for decode')
        
        return decoded_data


    def noisy_count(self, data, rho):
        unique_value, count = np.unique(data, return_counts=True)
        count = count + np.sqrt(1/(2 * rho)) * np.random.randn(len(count)) 
        
        return dict(zip(unique_value, count))



def calculate_rho_allocate(x_num, x_cat, num_encode_type, num_threshold=100, cat_threshold=100):
    # by default, we regard label (y) as variable which doesn't need any preprocess
    # the output is the portion that each preprocess would be allocated

    if x_num is not None and x_cat is not None: 
        num_unique_count = np.array([len(set(x_num[:,i])) for i in range(x_num.shape[1])])
        cat_unique_count = np.array([len(set(x_cat[:,i])) for i in range(x_cat.shape[1])])
        num_apply = (num_unique_count >= num_threshold)
        cat_apply = (cat_unique_count >= cat_threshold)

        if num_encode_type in ['privtree', 'dawa']:
            if sum(num_apply) > 0 and sum(cat_apply) > 0:
                num_rho_divide = sum(num_apply)/(sum(num_apply) + sum(cat_apply))
                cat_rho_divide = sum(cat_apply)/(sum(num_apply) + sum(cat_apply)) 
                return num_rho_divide, cat_rho_divide 
            elif sum(num_apply) > 0 and sum(cat_apply) == 0:
                return 1, 0
            elif sum(num_apply) == 0 and sum(cat_apply) > 0:
                return 0, 1 
            else:
                return 0, 0 
        else:
            if sum(cat_apply) > 0:
                return 0, 1
            else:
                return 0, 0
    
    elif x_num is None and x_cat is not None:
        cat_unique_count = np.array([len(set(x_cat[:,i])) for i in range(x_cat.shape[1])])
        cat_apply = (cat_unique_count >= cat_threshold) 

        if sum(cat_apply) > 0:
            return 0, 1
        else:
            return 0, 0
    
    elif x_cat is None and x_num is not None:
        num_unique_count = np.array([len(set(x_num[:,i])) for i in range(x_num.shape[1])])
        num_apply = (num_unique_count >= num_threshold)
        if sum(num_apply) > 0:
            return 1, 0 
        else:
            return 0, 0
    
    else: 
        return 0, 0
    