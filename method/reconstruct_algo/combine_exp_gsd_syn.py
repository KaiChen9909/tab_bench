############################################################################

# This file aims to use privsyn marginal selection and gsd synthesizers
# The marginal selection is from PrivSyn

############################################################################
import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import argparse
import os 
import json
import math
import sklearn.preprocessing

from method.private_gsd.utils.utils_data import Dataset, Domain
from method.private_gsd.stats import Marginals, ChainedStatistics
from method.private_gsd.models import GSD 
from method.private_gsd.utils.cdp2adp import cdp_rho
from jax.random import PRNGKey
from method.privsyn.PrivSyn.privsyn import PrivSyn


def prepare_domain(domain):
    domain_new = {}
    for k,v in domain.items():
        if k.split('_')[0] == 'num':
            domain_new[k] = 1
        else:
            domain_new[k] = v
    return domain_new

def prepare_df(df):
    num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
    
    num_idx = []
    for i in range(df.shape[1]):
        if df.columns[i].split('_')[0] == 'num':
            num_idx.append(i)
    
    df.iloc[:, num_idx] = num_encoder.fit_transform(df.iloc[:, num_idx])

    return df, num_encoder, num_idx

def gsd_syn_main(args, df, domain, rho, **kwargs): 
    domain_new = Domain.fromdict(domain)
    # df, encoder, num_idx  = prepare_df(df)
    data = Dataset(df, domain_new)

    marginals = PrivSyn.two_way_marginal_selection(data.df, data.domain.config, 0.1*rho, 0.8*rho)
    marginal_module1 = Marginals.get_all_kway_combinations(data.domain, k=1, bins=[2, 4, 8, 16, 32])
    marginal_module2 = Marginals.get_outside_kway_combinations(marginals, data.domain, k=2, bins=[2, 4, 8, 16, 32])
    stat_module = ChainedStatistics([marginal_module1, marginal_module2], rho_allocate=[0.1, 0.9])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()

    algo = GSD(
        domain=data.domain,
        print_progress=True,
        stop_early=True,
        # num_generations=20000,
        population_size_muta=50,
        population_size_cross=50,
        # data_size = df.shape[0]
        args=args
        # ,num_encoder=encoder,
        # num_idx=num_idx
    ) 

    key = PRNGKey(0)
    algo.zcdp_syn_init(key, stat_module, 0.9*rho)

    return {'gsd_syn_generator': algo}