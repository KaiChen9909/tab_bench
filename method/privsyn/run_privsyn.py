import logging
#import mkl
import os
import json
import sys
target_path="./"
sys.path.append(target_path)

from method.privsyn.PrivSyn.privsyn import PrivSyn
from method.privsyn.parameter_parser import parameter_parser
from method.privsyn.lib_preprocess.preprocess_network import PreprocessNetwork
from evaluator.eval_seeds import eval_seeds
# from evaluator.eval_query import eval_query 
# from evaluator.eval_fid import eval_fid


def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def add_default_params(args):
    args.dataset_name = args.dataset
    args.is_cal_marginals = True 
    args.is_cal_depend = True
    args.is_combine = True 
    args.marg_add_sensitivity = 1.0
    args.marg_sel_threshold = 20000
    args.non_negativity = "N3"
    args.consist_iterations = 501
    args.initialize_method = "singleton"
    args.update_method = "S5"
    args.append = True 
    args.sep_syn = False 
    args.update_rate_method = "U4"
    args.update_rate_initial = 1.0
    args.update_iterations = 50

    return args

def privsyn_main(args, df, domain, rho, **kwargs):
    config_logger()

    args = vars(add_default_params(args))
    privsyn_generator = PrivSyn(args, df, domain, rho) 

    return {"privsyn_generator": privsyn_generator}



if __name__ == "__main__":
    args = parameter_parser()
    
    privsyn_main(args)
