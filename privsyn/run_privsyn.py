import logging
#import mkl
import os
import json
import sys
target_path="./"
sys.path.append(target_path)

from privsyn.PrivSyn.privsyn import PrivSyn
from privsyn.parameter_parser import parameter_parser
from privsyn.lib_preprocess.preprocess_network import PreprocessNetwork
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
    # config the logger
    config_logger()
    # os.chdir("../../")

    # dataset_name = args.dataset
    # preprocess = PreprocessNetwork(dataset_name)
    # preprocess.load_data()
    # eps_num, eps_cat = preprocess.build_mapping(args['epsilon'], args['num_prep'], args['rare_threshold'])
    # preprocess.save_data(dataset_name, dataset_name + '_mapping')
    # preprocess.reverse_mapping()
    # preprocess.save_data_csv(dataset_name + '_syn_trivial.csv')

    args = vars(add_default_params(args))
    privsyn_generator = PrivSyn(args, df, domain, rho) 

    return {"privsyn_generator": privsyn_generator}

    # synthesized_filename = '_'.join((dataset_name, str(args['epsilon'])))
    # mapping_filename = dataset_name + '_mapping'

    # postprocess = PreprocessNetwork(dataset_name)
    # postprocess.reverse_mapping_from_files(synthesized_filename, mapping_filename)
    # postprocess.save_data_csv(synthesized_filename + '.csv')









    # this is the evaluation part for experiment
    # with open(f'data/{dataset_name}/info.json', 'r') as file:
    #     info = json.load(file)
    # n_classes = info['n_classes']
    # file.close()

    # eval_config = {
    #     'parent_dir': f'privsyn/exp/{dataset_name}_{eps}/',
    #     'real_data_path': f'data/{dataset_name}/',
    #     'model_params':{'num_classes': n_classes},
    #     'synthesized_filename': synthesized_filename,
    #     'mapping_filename': mapping_filename,
    #     'sample': {
    #             'seed': 0,
    #             'sample_num': privsyn_method.args['num_synthesize_records']
    #         },
    #     'eval':{
    #             'T':{
    #                 'seed': 0,
    #                 'normalization': "quantile",
    #                 'num_nan_policy': None,
    #                 'cat_nan_policy': None,
    #                 'cat_min_frequency': None,
    #                 'cat_encoding': "one-hot",
    #                 'y_policy': "default"
    #             }
    #         }
    # } 

    # print('*'*100)
    # print('Evaluation step')

    # eval_seeds(
    #     eval_config,
    #     sampling_method = 'privsyn',
    #     device = args['device'],
    #     privsyn_method = privsyn_method,
    #     privsyn_postprocess = postprocess
    # )


if __name__ == "__main__":
    args = parameter_parser()
    
    privsyn_main(args)
