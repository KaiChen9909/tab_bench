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
from scipy.optimize import bisect
from evaluator.eval_seeds import eval_seeds
from method.AIM.cdp2adp import cdp_rho

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
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    def score(cl):
        return sum(
            weights[workload_cl] * len(set(cl) & set(workload_cl))
            for workload_cl in workload_cliques
        )

    return {cl: score(cl) for cl in downward_closure(workload_cliques)}


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


class AIM(Mechanism):
    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        rho=None,
        bounded=None,
        rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
    ):
        if rho is None:
            super(AIM, self).__init__(epsilon, delta, bounded)
        else:
            self.rho = rho 
            self.prng = np.random
            self.bouned = bounded
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl] # candidates is filtered workload
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def error(self, model, answers, cl):
        x = answers[cl]
        xest = model.project(cl).datavector()
        errors = np.linalg.norm(x - xest, 1)
        return errors

    def run(self, data, workload, initial_cliques=None):
        rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload) # all possible subset of two-way marginals
    
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        if not initial_cliques:
            initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]  # use one-way marginals

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        marginal_dict = {}
        print("Initial Sigma", sigma)
        rho_used = len(oneway) * 0.5 / sigma ** 2
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

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(candidates, self.model, size_limit)

            cl = self.worst_approximated(
                small_candidates, answers, self.model, epsilon, sigma
            )

            # if t == 1:
                # cl = ('num_attr_8', 'num_attr_12')
            # print('cl:', cl)
            # print('start: ', self.error(self.model, answers, cl))

            n = data.domain.size(cl)
            Q = Identity(n)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = self.model.project(cl).datavector()

            if cl not in marginal_dict.keys():
                marginal_dict[cl] = 1
            else:
                marginal_dict[cl] += 1

            self.model = engine.estimate(measurements)
            w = self.model.project(cl).datavector()
            # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma", sigma / 2)
                sigma /= 2
                epsilon *= 2
            
            # print('end:', self.error(self.model, answers, cl))
        print('Selected marginals:', marginal_dict)
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

def add_default_params(args):
    args.max_model_size = 100
    args.max_iters = 1000
    args.degree = 2
    args.num_marginals = None 
    args.max_cells = 250000
    return args


def aim_main(args, df, domain, rho, **kwargs):
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
    mech = AIM(
        rho = rho,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    mech.run(data, workload)

    return {'aim_generator': mech}


# def default_params():
#     """
#     Return default parameters to run this program

#     :returns: a dictionary of default parameter settings for each command line argument
#     """
#     params = {}
#     params["dataset"] = "PUMSincome_period"
#     params["device"] = "cuda:0"
#     params["epsilon"] = 1.0
#     params["delta"] = 1e-5
#     # params["noise"] = "laplace"
#     params["max_model_size"] = 80
#     params["max_iters"] = 1000
#     params["degree"] = 2
#     params["num_marginals"] = None
#     params["max_cells"] = 10000

#     return params


# if __name__ == "__main__":

#     description = ""
#     formatter = argparse.ArgumentDefaultsHelpFormatter
#     parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
#     parser.add_argument("--dataset", help="dataset to use")
#     parser.add_argument("--device", help="device to use")
#     parser.add_argument("--epsilon", type=float, help="privacy parameter")
#     parser.add_argument("--delta", type=float, help="privacy parameter")
#     parser.add_argument(
#         "--max_model_size", type=float, help="maximum size (in megabytes) of model"
#     )
#     parser.add_argument("--max_iters", type=int, help="maximum number of iterations")
#     parser.add_argument("--degree", type=int, help="degree of marginals in workload")
#     parser.add_argument(
#         "--num_marginals", type=int, help="number of marginals in workload"
#     )
#     parser.add_argument(
#         "--max_cells",
#         type=int,
#         help="maximum number of cells for marginals in workload",
#     )
#     parser.add_argument("--save", type=str, help="path to save synthetic data")
#     parser.add_argument("--no_eval", action="store_true", default=False)  
#     parser.add_argument("--num_preprocess", type=str, default='privtree')
#     parser.add_argument("--rare_threshold", type=float, default=0.005)

#     parser.set_defaults(**default_params())
#     args = parser.parse_args()

#     os.makedirs(f'AIM/exp/{args.dataset}_{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}', exist_ok=True) 
#     parent_dir = f'AIM/exp/{args.dataset}_{args.epsilon}_{args.num_preprocess}_{args.rare_threshold}'
#     data_path = f'data/{args.dataset}/'

#     total_rho = cdp_rho(args.epsilon, args.delta)

#     data, num_rho, cat_rho = Dataset.load(data_path, total_rho, args.num_preprocess, args.rare_threshold)
#     preprocess_rho = (num_rho + cat_rho) * 0.1 * total_rho

#     workload = list(itertools.combinations(data.domain, args.degree))
#     workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
#     if args.num_marginals is not None:
#         workload = [
#             workload[i]
#             for i in np.random.choice(len(workload), args.num_marginals, replace=False)
#         ]

#     workload = [(cl, 1.0) for cl in workload]
#     mech = AIM(
#         args.epsilon,
#         args.delta,
#         preprocess_rho = preprocess_rho,
#         max_model_size=args.max_model_size,
#         max_iters=args.max_iters,
#     )
#     _, marginal_dict = mech.run(data, workload)

#     ############################### evaluation step ##############################

#     if not args.no_eval: 
#         with open(f'data/{args.dataset}/info.json', 'r') as file:
#             data_info = json.load(file)
#         config = {
#                 'parent_dir': parent_dir,
#                 'real_data_path': f'data/{args.dataset}/',
#                 'model_params':{'num_classes': data_info['n_classes']},
#                 'sample': {'seed': 0, 'sample_num': data_info['train_size']}
#             }
#         aim_dict = {
#             "num_encoder": data.num_encoder,
#             "cat_encoder": data.cat_encoder,
#             "num_col": data.num_col,
#             "cat_col": data.cat_col
#         }
#         with open(os.path.join(parent_dir, 'config.json'), 'w', encoding = 'utf-8') as file: 
#             json.dump(config, file)
#         print(aim_dict)

#         # evaluator function
#         eval_seeds(
#             raw_config = config,
#             n_seeds = 1,
#             n_datasets = 5,
#             device = args.device,
#             sampling_method = 'aim',
#             aim_generator = mech,
#             aim_dict = aim_dict
#         )