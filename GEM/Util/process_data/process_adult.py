import os
import csv
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import pdb

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain',
        'capital-loss', 'hours-per-week', 'native-country', 'income>50K']
cols_cont = ['fnlwgt', 'age', 'capital-gain', 'capital-loss', 'hours-per-week']
cols_categorical = [col for col in cols if col not in cols_cont]

df_train = pd.read_csv('./Datasets/adult/adult.data', names=cols)
df_test = pd.read_csv('./Datasets/adult/adult.test', names=cols).loc[1:]
df_test.loc[:, 'income>50K'] = df_test['income>50K'].apply(lambda x: x[:-1])
df = pd.concat([df_train, df_test]).drop_duplicates().reset_index(drop=True)

df.loc[:, 'fnlwgt'] = np.around(df['fnlwgt'] / df['fnlwgt'].max() * 100)

# convert some continuous attributes to categorical
bin_size = 1000
df.loc[df['capital-loss'] == 0, 'capital-loss'] = -1 * bin_size
df.loc[:, 'capital-loss'] //= 1000

df.loc[df['capital-gain'] == 0, 'capital-gain'] = -1 * bin_size
df.loc[df['capital-gain'] >= 50000, 'capital-gain'] = -2 * bin_size
df.loc[df['capital-gain'] >= 30000, 'capital-gain'] = -3 * bin_size
df.loc[df['capital-gain'] >= 20000, 'capital-gain'] = -4 * bin_size
df.loc[df['capital-gain'] >= 15000, 'capital-gain'] = -5 * bin_size
df.loc[df['capital-gain'] >= 10000, 'capital-gain'] = -6 * bin_size
df.loc[:, 'capital-gain'] //= 1000

df.loc[:, 'hours-per-week'] //= 10

# quantize cols
df.loc[:, 'age_10'] = df['age'].astype(int) // 10
cols_cont.append('age_10')

df.loc[:, cols_cont] = df[cols_cont].values.astype(int)
df.loc[:, cols_categorical] = df[cols_categorical].values.astype(str)

df_final = []
for wgt in df['fnlwgt'].unique():
    mask = df['fnlwgt'] == wgt
    rows = df.loc[mask]
    # df_wgt = pd.concat([rows]*wgt)
    df_wgt = pd.concat([rows]*1)
    df_final.append(df_wgt)
df_final = pd.concat(df_final)
del df_final['fnlwgt']
cols_cont = cols_cont[1:]

mappings = {}
domain = {}
for col in cols_categorical:
    enc = LabelEncoder()
    encoded = enc.fit_transform(df_final[col])
    df_final.loc[:, col] = encoded
    mapping = enc.classes_
    mappings[col] = mapping
    domain[col] = len(mapping)

for col in cols_cont:
    df_final.loc[:, col] -= np.minimum(df_final[col].min(), 0)
    domain[col] = int(df_final[col].max() + 1)

csv_path = "Datasets/adult/adult.csv"
domain_path = "Datasets/adult/adult-domain.json"

# save
df_final.to_csv(csv_path)
with open(domain_path, 'w') as f:
    json.dump(domain, f)