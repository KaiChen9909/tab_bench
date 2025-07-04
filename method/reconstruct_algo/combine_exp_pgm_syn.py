import os 
import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import pandas as pd
import argparse
import itertools
import json

from method.AIM.mbi.Dataset import Dataset
from method.AIM.mbi.inference import FactoredInference
from method.AIM.mbi.graphical_model import GraphicalModel
from method.AIM.mbi.Domain import Domain
from method.AIM.mbi.Factor import Factor
from method.AIM.mechanism import Mechanism
from collections import defaultdict
from method.AIM.mbi.matrix import Identity
from collections import defaultdict
from method.privsyn.PrivSyn.privsyn import PrivSyn

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2 ** 20

# def hypothetical_model_size(domain, cliques):
#     jtree, _ = junction_tree.make_junction_tree(domain, cliques)
#     maximal_cliques = junction_tree.maximal_cliques(jtree)
#     cells = sum(domain.size(cl) for cl in maximal_cliques)
#     size_mb = cells * 8 / 2**20
#     return size_mb


def compile_workload(workload):
    # this is different from aim's work, since other selection methods allocate same budget to all marginals
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    return {cl: 1 for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
            hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class PGM_Generator(Mechanism):
    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        rho=None,
        bounded=None,
        # rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
    ):
        if rho is None:
            super().__init__(epsilon, delta, bounded)
        else:
            self.rho = rho 
            self.prng = np.random
            self.bouned = bounded
        # self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros


    def run_pgm(self, data, k, workload):
        # rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload) # all possible subset of two-way marginals
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        measurements = []

        zeros = self.structural_zeros
        engine = FactoredInference(
            data.domain, iters=self.max_iters, warm_start=True, structural_zeros=zeros
        )

        sigma = np.sqrt(len(data.domain.attrs)/(0.2*self.rho))
        for attr in data.domain.attrs:
            cl = (attr, )
            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))

        cl_set = PrivSyn.two_way_marginal_selection(data.df, data.domain.config, 0.1*self.rho, 0.8*self.rho) # selection use 0.1rho
        print("Selected marginals:", cl_set)
        
        sigma = np.sqrt(len(cl_set)/(1.6*self.rho)) # use 0.8rho 
        
        for i in range(len(cl_set)):
            cl = cl_set[i] 

            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))

        print("Start model construction")
        engine.iters = self.max_iters
        self.model = engine.estimate(measurements) 
        print("Finish model construction")
        
        return self.model, cl_set

    def syn_data(
            self, 
            num_synth_rows, 
            path = None,
            preprocesser = None
        ):
        synth = self.model.synthetic_data(rows=num_synth_rows)
        if path is None:
            print('This is the raw data needed to be decoded')
            return synth
        else:
            synth.save_data_npy(path, preprocesser)
            return None



def add_default_params(args):
    args.max_model_size = 100
    args.max_iters = 1000
    args.degree = 2
    args.num_marginals = None 
    args.max_cells = 100000
    args.k = 5
    return args 


def pgm_syn_main(args, df, domain, rho, **kwargs):
    args = add_default_params(args)
    domain = Domain(domain.keys(), domain.values())
    data = Dataset(df, domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [
            workload[i]
            for i in np.random.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    mech = PGM_Generator(
        rho = rho,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    mech.run_pgm(data, args.k, workload)

    return {'pgm_syn_generator': mech}