#######################################################################

# This file will use Privsyn + PGM and Gumbel adaptive + PGM

#######################################################################

import os 
import sys
target_path="./"
sys.path.append(target_path)
import numpy as np
import pandas as pd
import argparse
import itertools
import json
import random
from tqdm import tqdm

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


def compile_workload(workload):
    # this is different from aim's work, since other selection methods allocate same budget to all marginals
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    return {cl: 1 for cl in downward_closure(workload_cliques)}


def compile_workload_two_way(workload):
    # this is different from aim's work, since other selection methods allocate same budget to all marginals
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    return {cl: 1 for cl in workload_cliques}


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




def double_filter(candidates, unselect):
    return {k:v for k,v in candidates.items() if k not in unselect}

class Gumbel_Generator(Mechanism):
    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        rho=None,
        n_records = None,
        bounded=None,
        # rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
        args=None
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
        self.args = args
        self.n_records = n_records

    def gumbel_mechanism(self, k, qualities, rho, base_measure=None, sizes=None, sigma_measure=None):
        if isinstance(qualities, dict):
            # import pandas as pd
            # print(pd.Series(list(qualities.values()), list(qualities.keys())).sort_values().tail())
            keys = list(qualities.keys())
            qualities = np.array([qualities[key] for key in keys])
            if base_measure is not None:
                base_measure = np.log([base_measure[key] for key in keys])
        else:
            qualities = np.array(qualities)
            keys = np.arange(qualities.size)

        """ Sample a candidate from the permute-and-flip mechanism """
        if sizes is not None:
            qualities = qualities - np.sqrt(2/np.pi)* sigma_measure * np.array([v for k,v in sizes.items()])/ self.n_records

        noise_vector = np.random.gumbel(loc=0.0, scale=k/(self.n_records * np.sqrt(2*rho)), size=len(qualities))
        noisy_qualities = qualities + noise_vector

        if len(noisy_qualities) >= k:
            selected_idx = np.argpartition(noisy_qualities, -k)[-k:]
            return [keys[i] for i in selected_idx]
        else:
            return keys
    
    def decide_selected(self, cl):
        for attr in cl:
            if self.selected[attr] is True:
                return True
        return False
    
    def worst_approximated_gumbel(self, candidates, answers, model, k, rho, sigma_measure=None):
        errors = {}
        sizes = None if sigma_measure is None else {}
        for cl in candidates:
            x = answers[cl]
            if self.decide_selected(cl):
                xest = model.project(cl).datavector()
            else:
                if max(model.project(cl).datavector()) < 1: 
                    assert (
                        sum(model.project(cl).datavector()) > 0.99
                        and 
                        sum(model.project(cl).datavector()) < 1.01
                    )
                    xest = [x*self.n_records for x in model.project(cl).datavector()]
                else:
                    xest = model.project(cl).datavector()

            errors[cl] = np.linalg.norm(x - xest, 1)/self.n_records
            if sizes is not None:
                sizes[cl] = model.domain.size(cl)

        # print('errors:', errors)
        # raise 'debug'
        return self.gumbel_mechanism(k, errors, rho, sizes=sizes, sigma_measure=sigma_measure)

    def decide_T(self):
        if self.args.dataset == 'loan':
            return 10
        elif self.args.dataset == 'higgs-small':
            return 7
        elif self.args.dataset == 'bank':
            return 6
        elif self.args.dataset == 'PUMSemploy_period_noage':
            return 6
        elif self.args.dataset == 'PUMSincome_period':
            return 4
        else:
            raise 'invalid dataset'
        
    def error(self, model, answers, cl):
        x = answers[cl]
        xest = model.project(cl).datavector()
        errors = np.linalg.norm(x - xest, 1)/self.n_records
        return errors

    def run_gumbel(self, data, k, workload):
        # rounds = self.rounds or 16 * len(data.domain)
        T = self.decide_T()
        candidates = compile_workload_two_way(workload) # all possible subset of two-way marginals
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        print(f"Total rounds: {T}, number of marginal candidates: {len(candidates)}")

        measurements = []
        marginal_dict = {}

        # initialize model
        zeros = self.structural_zeros
        engine = FactoredInference(
            data.domain, iters=self.max_iters, warm_start=True, structural_zeros=zeros
        )
        self.model = engine.estimate(measurements)

        self.selected = {k: False for k in data.df.columns}
        unselect = []

        # adaptive selection
        for t in tqdm(range(T)):
            size_limit = self.max_model_size * (t+1) / T
            # size_limit = self.max_model_size
            small_candidates = filter_candidates(candidates, self.model, size_limit)
            small_candidates = double_filter(small_candidates, unselect)
            sigma = np.sqrt(k*T/(1.0 *self.rho)) # two way marginal use 0.5rho

            cl_set = self.worst_approximated_gumbel(
                small_candidates, answers, self.model, k, (0.5*self.rho)/T
                # , sigma_measure=sigma
            ) # gumbel use 0.5rho

            for cl in cl_set: 
                for x in cl:
                    self.selected[x] = True

                n = data.domain.size(cl)
                x = data.project(cl).datavector()
                y = x + self.gaussian_noise(sigma, n)
                I = Identity(n)
                measurements.append((I, y, sigma, cl))

                if cl not in marginal_dict.keys():
                    marginal_dict[cl] = 1
                else:
                    marginal_dict[cl] += 1

                self.model = engine.estimate(measurements)
                unselect.append(cl) # This step is necessary for gumbel select, unless it will continuously choose one marginal

        print(marginal_dict)

        print("Start model construction")
        engine.iters = self.max_iters
        self.model = engine.estimate(measurements) 
        print("Finish model construction")

        return self.model, marginal_dict

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


class Privsyn_Generator(Mechanism):
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
    
    def worst_approximated_gumbel(self, candidates, answers, model, k, rho):
        errors = {}
        for cl in candidates:
            x = answers[cl]
            xest = model.project(cl).datavector()
            errors[cl] = np.linalg.norm(x - xest, 1)

        return self.gumbel_mechanism(k, errors, rho)


    def run_privsyn(self, data, k, workload):
        # rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload) # all possible subset of two-way marginals
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        measurements = []
        marginal_dict = {}

        # initialize model by one-way marginals
        initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]
        sigma = np.sqrt(len(initial_cliques)/(0.2 *self.rho))
        for cl in initial_cliques:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, x.size)
            I = Identity(y.size)
            measurements.append((I, y, sigma, cl))
            if cl not in marginal_dict.keys():
                marginal_dict[cl] = 1
            else:
                marginal_dict[cl] += 1

        zeros = self.structural_zeros
        engine = FactoredInference(
            data.domain, iters=self.max_iters, warm_start=True, structural_zeros=zeros
        )
        self.model = engine.estimate(measurements)

        cl_set = PrivSyn.two_way_marginal_selection(data.df, data.domain.config, 0.1*self.rho, 0.8*self.rho) # selection use 0.1rho
        # cl_selected = {key: False for key in cl_set.keys()}
        random.shuffle(cl_set)
        print("Selected marginals:", cl_set)
        
        # model construction
        sigma = np.sqrt(len(cl_set)/(1.6*self.rho))
        for cl in cl_set:
            # cl_selected[cl] = True
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
    args.k = 3
    return args


def gumbel_select_main(args, df, domain, rho, **kwargs):
    args = add_default_params(args)
    domain = Domain(domain.keys(), domain.values())
    n_records = df.shape[0]
    data = Dataset(df, domain)

    # workload0 = list(itertools.combinations(data.domain, 1))
    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [
            workload[i]
            for i in np.random.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    mech = Gumbel_Generator(
        rho = rho,
        n_records = n_records,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
        args=args
    )
    mech.run_gumbel(data, args.k, workload)

    return {'gumbel_select_generator': mech} 


def privsyn_select_main(args, df, domain, rho, **kwargs):
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
    mech = Privsyn_Generator(
        rho = rho,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    mech.run_privsyn(data, args.k, workload)

    return {'privsyn_select_generator': mech}