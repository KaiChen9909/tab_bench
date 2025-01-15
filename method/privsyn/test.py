import os
import logging
import pickle
import json
import ssl
import zipfile
import os.path as osp

import numpy as np
import pandas as pd

import config
from lib_dataset.dataset import Dataset
from lib_dataset.domain import Domain
from lib_preprocess.preprocess_network import PreprocessNetwork


def analyze_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Display basic information about the dataset
    print("Basic Information:")
    print(data.info())

    # Displaying the first few rows of the dataset
    print("\nFirst 5 Rows:")
    print(data.head())

    # Generating descriptive statistics for the dataset
    print("\nDescriptive Statistics:")
    print(data.describe(include='all'))

    # Analyzing the distribution of values for each field
    print("\nValue Counts for Each Field:")
    for column in data.columns:
        print(f"\nColumn: {column}")
        print(data[column].value_counts())



def main():
    os.chdir("../../")

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    #data_df = pd.read_pickle(config.PROCESSED_DATA_PATH + 'ton')
    #print(data_df.df.head())
    #print(data_df.domain.config)
    #mapping = pd.read_pickle(config.PROCESSED_DATA_PATH + 'ton_10000_mapping')
    #print(mapping)
    #p = pd.read_pickle(config.DEPENDENCY_PATH + 'ton_10000')
    #print(p)
    #p = pd.read_pickle(config.MARGINAL_PATH + 'ton_10000_2.0')
    #print(p)
    #preprocess = PreprocessTon()
    #preprocess.reverse_mapping_from_files('ton_10.0', 'ton_mapping')
    #preprocess.save_data_csv('ton_10.0_syn.csv')


    # Replace 'your_dataset.csv' with the path to your dataset
    
    #analyze_dataset(config.RAW_DATA_PATH+'ton.csv')
    #analyze_dataset(config.SYNTHESIZED_RECORDS_PATH+'ton_2.0_syn.csv')
    #analyze_dataset(config.SYNTHESIZED_RECORDS_PATH+'ton_10.0_syn.csv')
    analyze_dataset(config.SYNTHESIZED_RECORDS_PATH+'ton_2.0.csv')

if __name__ == "__main__":
    main()