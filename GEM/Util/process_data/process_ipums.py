import os
import csv
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from xml.dom import minidom
from sklearn.preprocessing import LabelEncoder

import pdb

# python Util/process_data/process_ipums.py --fn acs_2010_1yr acs_2014_1yr acs_2018_1yr

def row_generator(datapath, ddipath):
    ''' Maps each line of the data file to the variables and values
        it represents '''
    # get mapping
    pmap = pos_map(ddipath)
    f = open(datapath, 'r')
    for line in f:
        # apply mapping
        row = {}
        for var in pmap.keys():
            start = pmap[var]['spos']
            end = pmap[var]['epos']
            dec = pmap[var]['dec']
            if dec:
                mid = end - dec
                row[var] = line[start:mid] + '.' + line[mid:end]
            else:
                row[var] = line[start : end]
        # yield mapping
        yield row

def pos_map(ddipath):
    ''' Returns a dictionary mapping the variable names to their positions
        and decimal places in the data file '''
    m = minidom.parse(ddipath)    
    vmap = {}
    varNodes = m.getElementsByTagName('var')
    for varNode in varNodes:
        locNode = varNode.getElementsByTagName('location')[0]
        name = varNode.attributes.getNamedItem('ID').value
        vmap[name] = {
            'spos' : int(locNode.attributes.getNamedItem('StartPos').value) - 1,
            'epos' : int(locNode.attributes.getNamedItem('EndPos').value),
            'dec' : int(varNode.attributes.getNamedItem('dcml').value)
            }
    return vmap

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./Datasets/acs')
    parser.add_argument('--fn', type=str, nargs='+')
    args = parser.parse_args()
    return args

def get_raw(root, fn):
    csv_path = '{}/{}_RAW.csv'.format(root, fn)

    datapath = '{}/{}.dat'.format(root, fn)
    ddipath = '{}/{}.xml'.format(root, fn)
    row_gen = row_generator(datapath=datapath, ddipath=ddipath)

    write_cols = True
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        for row in tqdm(row_gen):
            if write_cols:
                write_cols = False
                wr.writerow(list(row.keys()))
            wr.writerow(list(row.values()))


if __name__ == '__main__':
    args = get_args()

    df_all = []
    for fn in args.fn:
        csv_path_raw = '{}/{}_RAW.csv'.format(args.root, fn)
        if not os.path.exists(csv_path_raw):
            get_raw(args.root, fn)
        df = pd.read_csv(csv_path_raw)
        df['fn'] = fn
        df_all.append(df)

    columns = [list(df.columns.values) for df in df_all]
    columns = list(set.intersection(*map(set, columns)))
    for i in range(len(df_all)):
        df_all[i] = df_all[i][columns]
    df_all = pd.concat(df_all)

    # drop default columns that IPUMS automatically adds
    COLS_IGNORE = ['YEAR', 'SAMPLE', 'SERIAL', 'CBSERIAL', 'HHWT', 'CLUSTER', 'STRATA', 'GQ', 'PERNUM', 'PERWT']
    df_all.drop(columns=COLS_IGNORE, inplace=True)

    # drop detailed version of columns
    COLS_DETAILED = [col for col in df_all.columns if col[-1]=='D' and col[:-1] in df_all.columns]
    df_all.drop(columns=COLS_DETAILED, inplace=True)

    df_state_fips = pd.read_csv('{}/fips_mapping.csv'.format(args.root))
    df_state_fips = df_state_fips[['fips', 'state_abbr']]
    df_state_fips.columns = ['FIPS', 'STATE']

    df_all['STATEFIP'] = df_all['STATEFIP'].astype(int)
    df_all = pd.merge(df_state_fips, df_all, left_on=['FIPS'], right_on=['STATEFIP'])
    for col in ['FIPS', 'STATEFIP']:
        del df_all[col]

    # manual changes
    df_all['AGEORIG'] = df_all['AGE'].values
    df_all['AGE'] = df_all['AGE'] // 10

    print("Encoding columns")
    enc = LabelEncoder()    

    domain = {}
    mappings = {}
    for col in tqdm(df_all.columns):
        if col in ['fn', 'STATE']:
            continue
        encoded = enc.fit_transform(df_all[col])
        mapping = enc.classes_

        df_all.loc[:, col] = encoded
        mappings[col] = mapping
        domain[col] = len(mapping)

    for fn in args.fn:
        print("Saving {}...".format(fn))
        df = df_all[df_all['fn']==fn]
        df.reset_index(drop=True, inplace=True)
        del df['fn']

        csv_path = '{}/{}.csv'.format(args.root, fn)
        df.to_csv(csv_path, index=False)

        with open('{}/{}-domain.json'.format(args.root, fn), 'w') as f: 
            json.dump(domain, f)

        with open('{}/{}-mapping.pkl'.format(args.root, fn), 'wb') as f: 
            pickle.dump(mappings, f)