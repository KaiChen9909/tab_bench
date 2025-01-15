import os, sys
target_path="./"
sys.path.append(target_path)

import logging
import pickle
import json
import random
import ssl
import zipfile
import os.path as osp
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from collections import Counter
from preprocess_common.preprocess import * 

import numpy as np
import pandas as pd
from decimal import Decimal

# sys.path.append(os.getcwd())
import method.privsyn.config as config
from method.privsyn.lib_dataset.dataset import Dataset
from method.privsyn.lib_dataset.domain import Domain

import socket
import struct

from method.privsyn.parameter_parser import parameter_parser

class PreprocessNetwork:
    def __init__(self, csv_filename):
        self.logger = logging.getLogger("preprocess a network dataset")
        self.csv_filename = csv_filename
        self.shape = []
        
        for path in config.ALL_PATH:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Load the Field Types from JSON
        with open(os.path.join(config.TYPE_CONIFG_PATH, f'{csv_filename}_fields.json'), 'r') as file:
            field_config = json.load(file)
            self.field_types = field_config["field_types"]
            self.bin_sizes = field_config["bin_sizes"]
    
    def load_data(self):
        self.logger.info("loading data")
        #ZL: need to copy all datasets to raw_data_path
        x_num = x_cat = None
        num_col = cat_col = 0
        if os.path.exists(os.path.join(config.RAW_DATA_PATH, f'{self.csv_filename}/X_num_train.npy')):
            x_num = np.load(os.path.join(config.RAW_DATA_PATH, f'{self.csv_filename}/X_num_train.npy'), allow_pickle=True) 
            num_col = x_num.shape[1]
        if os.path.exists(os.path.join(config.RAW_DATA_PATH, f'{self.csv_filename}/X_cat_train.npy')):
            x_cat = np.load(os.path.join(config.RAW_DATA_PATH, f'{self.csv_filename}/X_cat_train.npy'), allow_pickle=True)
            cat_col = x_cat.shape[1]
        y = np.load(os.path.join(config.RAW_DATA_PATH, f'{self.csv_filename}/y_train.npy'), allow_pickle=True).reshape(-1,1)

        if x_num is None:
            data_arr = np.concatenate((x_cat,y), axis=1)
        elif x_cat is None:
            data_arr = np.concatenate((x_num,y), axis=1)
        else:
            data_arr = np.concatenate((x_num,x_cat,y), axis=1)

        self.num_col = [f'num_attr_{i}' for i in range(1, num_col+1)]
        self.cat_col = [f'cat_attr_{i}' for i in range(1, cat_col+1)]
        self.y_col = ['y_attr']
        self.col_name = self.num_col + self.cat_col + self.y_col
        self.df = pd.DataFrame(data_arr, columns = self.col_name)

        # with open(config.RAW_DATA_PATH + csv_filename, 'r') as file:
            # self.df = pd.read_csv(config.RAW_DATA_PATH + csv_filename, low_memory=False)

    # Function to bin IP addresses by subnet
    def bin_ip(self, ip_series, subnet_size):
        factor = 32 - subnet_size
        return ip_series.apply(lambda ip: int(ip) >> factor)

    # Exponential binning function
    def bin_exponential(self, value, base):
        if value <= 0:
            return 0
        return int(np.floor(np.log(value) / np.log(base)))

    def build_mapping_old(self):
        self.logger.info("build mapping")
        # Encode Categorical Fields and Handle Timestamp
        self.mappings = {}  # To store mappings of categorical fields
        self.shape = []
        for column in self.df.columns:
            if self.field_types[column] == 'categorical':
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                self.mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
                self.shape.append(len(le.classes_) + 1) 
            elif self.field_types[column] == 'binned-numerical':
                n_bins = min(len(set(self.df[column])), 500)
                bine = KBinsDiscretizer(
                    n_bins= n_bins,
                    encode='ordinal', 
                    strategy='uniform'
                )
                self.df[column] = bine.fit_transform(self.df[column])
                self.mappings[column] = bine
            elif self.field_types[column] == 'binned-ip':
                self.df[column] = self.bin_ip(self.df[column], self.bin_sizes[column])
                #self.shape.append(2 ** (32 - self.bin_sizes[column]))
                #self.shape.append(2 ** 32)
                self.shape.append(self.df[column].max() + 1)
            elif self.field_types[column] == 'binned-port':
                threshold = self.bin_sizes[column]
                bin_size = self.bin_sizes["port_bin_size"]
                self.df[column] = self.df[column].apply(lambda x: x if x < threshold else threshold + ((x - threshold) // bin_size))
                self.shape.append(self.df[column].max() + 1)
                #self.shape.append(65536)
            elif self.field_types[column] == 'binned_integer' or self.field_types[column] == 'timestamp':
                if self.field_types[column] == 'timestamp':
                    initial_timestamp = self.df[column].min()
                    self.mappings['initial_timestamp'] = initial_timestamp
                    self.df[column] -= initial_timestamp
                bin_size = self.bin_sizes.get(column, 1)
                bins = np.arange(0, self.df[column].max() + bin_size, bin_size)
                self.df[column] = np.digitize(self.df[column], bins, right=False)
                self.shape.append(len(bins)+1)
            elif self.field_types[column] in ['float-exponential', 'int-exponential']:
                # First, apply exponential binning
                base = self.bin_sizes[column]
                encoded_values = self.df[column].apply(lambda x: self.bin_exponential(float(x), base))

                # Find the minimum encoded value
                min_encoded_val = encoded_values.min()
                self.mappings[column + "_min_encoded_val"] = min_encoded_val

                # Shift the encoded values to ensure they are >= 0
                self.df[column] = encoded_values - min_encoded_val
                self.shape.append(self.df[column].max() + 1)

            self.logger.info(f"Encoded Column: {column}")
            self.logger.info(f"Number of Bins: {self.shape[-1]}")
            self.logger.info(f"Min Bin Value: {self.df[column].min()}")
            self.logger.info(f"Max Bin Value: {self.df[column].max()}")
            self.logger.info(f"Average Bin Value: {self.df[column].mean()}")


    def build_mapping(self, rho, num_prep, rare_threshold):
        self.logger.info("build mapping")
        self.mapping = {'num': None, 'cat': None} 
        X_num = np.array(self.df[self.num_col]) if len(self.num_col) > 0 else None
        X_cat = np.array(self.df[self.cat_col]) if len(self.cat_col) > 0 else None
        num_encoder = None
        cat_encoder = None

        num_rho, cat_rho = calculate_rho_allocate(X_num, X_cat, num_prep) 

        if X_num is not None:
            num_encoder = discretizer(num_prep, num_rho * 0.1 * rho)
            X_num = num_encoder.fit_transform(X_num)
        if X_cat is not None:
            cat_encoder = rare_merger(cat_rho * 0.1 * rho, rare_threshold=rare_threshold)
            X_cat = cat_encoder.fit_transform(X_cat) 
        
        self.df[self.num_col] = X_num 
        self.df[self.cat_col] = X_cat 
        self.mapping['num'] = num_encoder 
        self.mapping['cat'] = cat_encoder
        self.shape = [len(set(self.df[x])) for x in self.col_name]

        return num_rho, cat_rho


    def save_data(self, pickle_filename, mapping_filename):
        self.logger.info("saving data")
        domain = Domain(self.df.columns, self.shape)
        dataset = Dataset(self.df, domain)

        #Save the Encoded Dataset to a Pickle File
        with open(config.PROCESSED_DATA_PATH + pickle_filename, 'wb') as file:
            pickle.dump(dataset, open(config.PROCESSED_DATA_PATH + pickle_filename, 'wb'))
        #Save the Mappings to another Pickle File
        with open(config.PROCESSED_DATA_PATH + mapping_filename, 'wb') as file:
            pickle.dump(self.mappings, file)

        self.logger.info("saved data")


    def reverse_mapping_old(self):
        self.logger.info("reverse mapping")
        # self.df is the encoded version, after dpsyn, it should be updated before calling this function     
        # Decoding the dataset
        # Reversing the encoding for categorical fields
        for column, mapping in self.mappings.items():
            if column in self.field_types and self.field_types[column] == 'categorical':
                inv_map = {v: k for k, v in mapping.items()}
                self.df[column] = self.df[column].map(inv_map) 
            
            if column in self.field_types and self.field_types[column] == 'binned-numerical':
                self.df[column] = mapping.inverse_tranform(self.df[column])

        # Reversing the binning for IP addresses
        for column in self.df.columns:
            if self.field_types[column] == 'binned-ip':
                self.logger.info("binned-ip")
                self.logger.info("column name: " + column)
                subnet_size = self.bin_sizes[column]
                factor = 32 - subnet_size
                self.df[column] = self.df[column].apply(
                    lambda x: (x << factor) + random.randint(0, (1 << factor) - 1))

            # Reversing the binning for Ports
            elif self.field_types[column] == 'binned-port':
                self.logger.info("binned-port")
                self.logger.info("column name: " + column)
                threshold = self.bin_sizes[column]
                bin_size = self.bin_sizes["port_bin_size"]
                self.df[column] = self.df[column].apply(
                    lambda x: x if x < threshold else (threshold + (x - threshold) * bin_size) + random.randint(0, bin_size - 1))

            # Reversing the binning for timestamp
            elif self.field_types[column] == 'binned_integer' or self.field_types[column] == 'timestamp':  # For binned_integer fields and timestamps
                bin_size = self.bin_sizes[column]
                # For timestamp fields, reconstruct timestamps from bins
                if self.field_types[column] == 'timestamp':
                    self.logger.info("temestamp")
                    self.logger.info("column name: " + column)
                    initial_timestamp = self.mappings.get('initial_timestamp', 0)
                    # Randomly sample within each bin and add the initial timestamp
                    self.df[column] = self.df[column].apply(
                        lambda x: initial_timestamp + (x - 1) * bin_size + random.randint(0, bin_size - 1))
                else:
                    self.logger.info("binned-integer")
                    self.logger.info("column name: " + column)
                    # Randomly sample within each bin for other binned_integer fields
                    self.df[column] = self.df[column].apply(lambda x: (x - 1) * bin_size + random.randint(0, bin_size - 1))

            # Reversing the exponential binning
            elif self.field_types[column] in ['float-exponential', 'int-exponential']:
                min_encoded_val = self.mappings.get(column + "_min_encoded_val", 0)
                base = self.bin_sizes[column]

                if self.field_types[column] == 'float-exponential':
                    self.logger.info("float-exponential")
                    self.logger.info("column name: " + column)
                    # Sample a floating point value for float-exponential fields
                    self.df[column] = (self.df[column] + min_encoded_val).apply(
                        lambda x: (base ** x) + random.uniform(0, (base ** (x + 1)) - (base ** x)))
                else:
                    self.logger.info("int-exponential")
                    self.logger.info("column name: " + column)
                    # Sample an integer value for int-exponential fields
                    self.df[column] = (self.df[column] + min_encoded_val).apply(
                        lambda x: np.power(base, x) + random.randint(0, np.power(Decimal(base), x + 1)) - np.power(Decimal(base), x) - 1)

        #We don't syntheszing floating point
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(0, inplace=True)
        #for col in self.df.select_dtypes(include=['float64']):
        #    self.df[col] = self.df[col].astype(int)


    def reverse_mapping(self):
        self.logger.info("reverse mapping")
        if len(self.num_col) > 0:
            self.df[self.num_col] = self.mapping['num'].inverse_transform(np.array(self.df[self.num_col]))
        if len(self.cat_col) > 0:
            self.df[self.cat_col] = self.mapping['cat'].inverse_transform(np.array(self.df[self.cat_col]))
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(0, inplace=True)


    def reverse_mapping_from_files(self, pickle_filename, mapping_filename, record_path = None):
        if record_path is None:
            record_path = config.SYNTHESIZED_RECORDS_PATH
        with open(os.path.join(record_path, pickle_filename), 'rb') as file:
            ds = pickle.load(file)
            self.df = ds.df
        with open(config.PROCESSED_DATA_PATH + mapping_filename, 'rb') as file:
            self.mappings = pickle.load(file)
        self.reverse_mapping()
          

    def save_data_csv(self, csv_filename):
        self.logger.info("save df to csv file")
        # Save the Decoded Dataset to a CSV File
        with open(config.SYNTHESIZED_RECORDS_PATH + csv_filename, 'wb') as file:
            self.df.to_csv(config.SYNTHESIZED_RECORDS_PATH + csv_filename, index=False)


    def save_data_npy(self, save_path = None):
        cat_col_num = len(self.cat_col) 
        num_col_num = len(self.num_col) 
        print(f'numerical variables: {num_col_num}, categorical variables: {cat_col_num}, label: 1')
        self.logger.info("save df to npy file")

        if save_path is None: 
            save_path = config.SYNTHESIZED_RECORDS_PATH 

        if num_col_num > 0:
            x_num_syn = self.df.iloc[:, : num_col_num].to_numpy(dtype = np.float32)
            np.save(os.path.join(save_path, 'X_num_train.npy'), x_num_syn) 

        if cat_col_num > 0:
            x_cat_syn = self.df.iloc[:, num_col_num : -1].to_numpy(dtype = str)
            np.save(os.path.join(save_path, 'X_cat_train.npy'), x_cat_syn)
        
        y_syn = self.df.iloc[:, -1].to_numpy(dtype = np.float32).squeeze()
        np.save(os.path.join(save_path, 'y_train.npy'), y_syn)


def main(args):
    # config the logger
    
    # os.chdir("../../")

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)


    file_prefix = args['dataset_name']
    preprocess = PreprocessNetwork(file_prefix)
    preprocess.load_data()
    preprocess.build_mapping()
    preprocess.save_data(file_prefix, file_prefix + '_mapping')
    preprocess.reverse_mapping()
    preprocess.save_data_csv(file_prefix + '_syn_trivial.csv')

if __name__ == "__main__":
    args = parameter_parser()
    
    main(args)
