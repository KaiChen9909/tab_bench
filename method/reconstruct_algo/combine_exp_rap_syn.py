############################################################################

# This file aims to use privsyn marginal selection and rap synthesizers
# The marginal selection is from PrivSyn

############################################################################



import argparse
import os

from jax import devices

cpu = devices("cpu")[0]
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys

sys.path.append(os.path.dirname("./"))
"""
Use this environment:
https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-cuda-11-4-amazon-linux-2/
"""
import os
import time 
import numpy as np
import sklearn.preprocessing

from method.RAP.dataloading.data_functions.acs import *
from method.RAP.mechanisms.rap_pp import RAPpp, RAPppConfiguration
from method.RAP.modules.marginal_queries import MarginalQueryClass
# from RAP.modules.random_features_linear import RandomFeaturesLinearQueryClass
# from sklearn.linear_model import LogisticRegression
from method.RAP.dataloading.dataset import Dataset
from method.RAP.dataloading.domain import Domain
from method.RAP.dataloading.dataloading_util import get_upsampled_dataset_from_relaxed
from method.RAP.mechanisms.mechanism_base import BaseMechanism
from method.privsyn.PrivSyn.privsyn import PrivSyn


def add_default_params(args):
    args.seed = 0
    args.k = 2
    args.num_random_projections = 200000
    args.top_q = 5
    args.dp_select_epochs = 50
    args.algorithm = 'RAP++'
    return args


def get_syn_df(
        D_relaxed, domain, oversample_rate=1, seed=0
    ) -> Dataset:
        """
        This function takes as input a relaxed dataset matrix. Then creates a DataFrame in the original
            datas format.
        """
        D_prime_post_dataset = get_upsampled_dataset_from_relaxed(
            D_relaxed, domain, oversample=oversample_rate, seed=seed
        )
        return D_prime_post_dataset.df


class rap_outside_generator():
    def __init__(self, rho, seed, domain, mechanism, args):
        self.rho = rho 
        self.seed = seed 
        self.domain = domain 
        self.mechanism = mechanism
        self.args = args
    
    def syn(self, n_sample, preprocesser=None, parent_dir=None): 
        if self.args.dataset == 'loan':
            self.oversamples = 40 
        elif self.args.dataset == 'higgs-small':
            self.oversamples = 20
        else:
            self.oversamples = 10
        self.mechanism.outside_train(rho=self.rho, seed=self.seed, num_generated_points=n_sample//self.oversamples)
        D_prime_relaxed = self.mechanism.get_dprime()[:, :]
        D_prime_original_format_df = (
            get_syn_df(
                D_prime_relaxed, seed=self.seed, oversample_rate=self.oversamples, domain=self.domain
            )
        )
        if parent_dir is not None:
            preprocesser.reverse_data(D_prime_original_format_df, parent_dir)
        return D_prime_original_format_df

def get_one_way_marginal(domain):
    return [(x,) for x in domain.keys()]


def run_rap_syn_experiment(
        df,
        domain,
        rho: float,
        algorithm_seed: int,
        args
        # params: str,
        # ,save_sycn_data=True
    ):
    """Runs RAP++ and saves the relaxed synthetic data as .npy."""
    algo_seed = algorithm_seed

    domain_dict = domain
    # domain = Domain.fromdict(domain, targets = ['y_attr'])
    domain = Domain.fromdict(domain, targets = None)
    dataset = Dataset(df, domain)

    one_way_marginals = get_one_way_marginal(domain_dict)
    two_way_marginals = list(PrivSyn.two_way_marginal_selection(df, domain_dict, 0.1*rho, 0.8*rho)) 
    print(two_way_marginals)
    marginals = [one_way_marginals, two_way_marginals]

    rap_args1 = RAPppConfiguration(
        iterations=[3],
        sigmoid_doubles=[0],
        optimizer_learning_rate=[0.005],
        top_q=1,
        get_dp_select_epochs=lambda domain: 1,
        get_privacy_budget_weight=lambda domain: 0.1,
        debug=False,
    )
    rap_args2 = RAPppConfiguration(
        iterations=[10 * df.shape[1]],
        sigmoid_doubles=[0],
        optimizer_learning_rate=[0.005],
        top_q=1,
        get_dp_select_epochs=lambda domain: 1,
        get_privacy_budget_weight=lambda domain: 0.8,
        debug=False,
    ) # for categorical
     
    rap_linear_projection = RAPpp(
        [
            rap_args1, 
            rap_args2
        ],
        [
            MarginalQueryClass(K=1, conditional=False),
            MarginalQueryClass(K=2, conditional=False),
        ],
        name=f"RAP(Marginal)",
    )  

    mechanism = rap_linear_projection

    mechanism.outside_initialize(
        dataset, algo_seed, marginals
    ) 

    generator = rap_outside_generator(
        rho = 0.9*rho,
        seed = algo_seed,
        domain = domain, 
        mechanism = mechanism,
        args=args
    )
    return generator



def rap_syn_main(args, df, domain, rho, parent_dir, **kwargs):
    args = add_default_params(args)

    generator = run_rap_syn_experiment(
        df,
        domain,
        rho=rho,
        args=args,
        algorithm_seed=args.seed
    )

    return {'RAP_syn_generator': generator}