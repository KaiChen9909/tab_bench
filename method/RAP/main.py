import argparse
import os

from jax import devices

cpu = devices("cpu")[0]
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import numpy as np
import sklearn.preprocessing

sys.path.append(os.path.dirname("./"))
"""
Use this environment:
https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-cuda-11-4-amazon-linux-2/
"""

import method.RAP.run as run
from method.RAP.dataloading.data_functions.acs import *
from method.RAP.mechanisms.rap_pp import RAPpp, RAPppConfiguration
from method.RAP.modules.marginal_queries import MarginalQueryClass
from method.RAP.modules.random_features_linear import RandomFeaturesLinearQueryClass

def add_default_params(args):
    args.seed = 0
    args.k = 2
    args.num_random_projections = 200000
    args.top_q = 5
    args.dp_select_epochs = 50
    args.algorithm = 'RAP++'
    return args
    # parser.add_argument(
    #     "--states", type=str, default="NY,CA", help="Load data from the state(s)"
    # )
    # parser.add_argument(
    #     "--targets",
    #     type=str,
    #     default="income,coverage",
    #     help="Load tasks from this state(s)",
    # )
    # parser.add_argument(
    #     "--multitask",
    #     action="store_true",
    #     help="Generate synthetic data that works for multiple tasks at once",
    # ) 
    
def prepare_domain(domain):
    domain_new = {}
    for k,v in domain.items():
        if k.split('_')[0] == 'num':
            domain_new[k] = 1
        else:
            domain_new[k] = v
    return domain_new



def count_domain_type(domain):
    contain_num = False 
    contain_cat = False
    for k,v in domain.items():
        if k.split('_')[0] == 'num':
            if v == 1:
                contain_num = True 
            else:
                contain_cat = True
        elif k.split('_')[0] == 'cat':
            contain_cat = True 
    
    return contain_num, contain_cat

def prepare_df(df):
    num_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
    
    num_idx = []
    for i in range(df.shape[1]):
        if df.columns[i].split('_')[0] == 'num':
            num_idx.append(i)
    
    df.iloc[:, num_idx] = num_encoder.fit_transform(df.iloc[:, num_idx]).astype(np.float16)

    return df, num_encoder, num_idx


def rap_main(args, df, domain, rho, parent_dir, **kwargs):
    args = add_default_params(args)

    domain = prepare_domain(domain)
    df, encoder, num_idx  = prepare_df(df)
    # df, encoder, num_idx = df, None, None 
    contain_num, contain_cat = count_domain_type(domain)

    print(f'contain_num: {contain_num}, contain_cat: {contain_cat}')


    rap_args = RAPppConfiguration(
        iterations=[1],
        sigmoid_doubles=[0],
        optimizer_learning_rate=[0.003],
        top_q=1,
        get_dp_select_epochs=lambda domain: len(domain.get_cat_cols()),
        get_privacy_budget_weight=lambda domain: len(domain.get_cat_cols()),
        debug=False,
    ) # for categorical

    rap_args2 = RAPppConfiguration(
        iterations=[3],
        sigmoid_0=[2],
        sigmoid_doubles=[10],
        optimizer_learning_rate=[0.006],
        top_q=args.top_q,
        get_dp_select_epochs=lambda domain: args.dp_select_epochs,
        get_privacy_budget_weight=lambda domain: len(domain.get_cont_cols()),
        debug=False,
    ) # for continuous

    if args.algorithm == "RAP++":
        if not contain_num: 
            rap_linear_projection = RAPpp(
                [rap_args],
                [
                    MarginalQueryClass(K=args.k),
                ],
                name=f"RAP(Marginal)",
            ) # if add conditional = False, then use 
        elif not contain_cat:
            rap_linear_projection = RAPpp(
                [rap_args2],
                [
                    RandomFeaturesLinearQueryClass(
                        num_random_projections=args.num_random_projections,
                        max_number_rows=2000,
                    ),
                ], #stat_module
                name=f"RAP(Halfspace)",
            ) 
        else:
            rap_linear_projection = RAPpp(
                [rap_args, rap_args2],
                [
                    MarginalQueryClass(K=args.k),
                    RandomFeaturesLinearQueryClass(
                        num_random_projections=args.num_random_projections,
                        max_number_rows=2000,
                    ),
                ], #stat_module
                name=f"RAP(Marginal&Halfspace)",
            )
    if args.algorithm == "RAP":
        rap_linear_projection = RAPpp(
            [rap_args],
            [
                MarginalQueryClass(K=args.k),
            ],
            name=f"RAP",
        )

    algorithm = rap_linear_projection

    override_errors_file = True
    saves_results = True


    generator = run.run_experiment(
        algorithm,
        df,
        domain,
        rho=rho,
        args=args,
        algorithm_seed=args.seed,
        num_encoder = encoder, 
        num_idx = num_idx
    )

    return {'RAP_generator': generator}
