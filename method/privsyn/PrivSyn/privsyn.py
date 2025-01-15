#####################################################################
#                                                                   #
#                The main procedure class of dpsyn,                 #
#               which implements algorithm 3 in paper               #
#                                                                   #
#####################################################################

import datetime
import logging
import math
import copy
import sys
target_path="./"
sys.path.append(target_path)

import numpy as np
import pandas as pd

from method.privsyn.lib_synthesize.converge_imp import sep_graph, clip_graph, append_attrs
from method.privsyn.lib_synthesize.update_config import UpdateConfig
from method.privsyn.lib_dataset.dataset import Dataset
from method.privsyn.lib_marginal.marg import Marginal
from method.privsyn.lib_marginal.consistent import Consistenter
from method.privsyn.lib_composition.advanced_composition import AdvancedComposition
from method.privsyn.lib_marginal.marg_determine import marginal_selection, marginal_combine
from method.privsyn.lib_marginal.filter import Filter
from method.privsyn.lib_dataset.data_store import DataStore
from method.privsyn.lib_dataset.domain import Domain

from functools import reduce

class PrivSyn():
    def __init__(self, args, df, domain, rho):
        ########################################### preprocess ###########################################
        self.logger = logging.getLogger('PrivSyn')
 
        self.no_filter = True if df is not None else False        
        
        self.one_way_marg_dict  = {}
        self.combined_marg_dict  = {}
        self.marg_dict  = {}
        self.singleton_key = [] 
        
        self.start_time = datetime.datetime.now()
        self.args = args
        
        self.dataset_name = args['dataset']
        self.total_rho = rho
        
        self.data_store = DataStore(self.args) # DataStore handles everything about data including reading and writing.
        self.load_data_from_df(df, domain)
        # self.load_data()

        self.num_records = self.original_dataset.df.shape[0]
        self.num_attributes = self.original_dataset.df.shape[1]
        
        self.logger.info("original dataset domain: %e" % (self.original_dataset.domain.size(),))
        self.privacy_budget_allocation()
        
        ########################################### main proceedure ###########################################
        print(list(self.original_dataset.domain.config.values()))

        self.construct_margs(mode = 'one_way')
        self.anonymize_margs(mode = 'one_way')
        
        self.filtered_dataset = self.filter() # one-way filter, if we have convey a df, this step will not work
        
        self.sel_marg_name = self.obtain_marginal_list(self.original_dataset)
        
        self.construct_margs(mode = 'combined')
        self.anonymize_margs(mode = 'combined')

        self.improving_convergence()
        self.consist_marginals(self.filtered_dataset.dataset_recode.domain, self.marg_dict)

        self.end_time = datetime.datetime.now()
        logger = logging.getLogger("excution completed")
        logger.info("model construction time: %s" % (self.end_time - self.start_time))
        #######################################################################################################
    
    @staticmethod
    def two_way_marginal_selection(df, domain, rho, rho1):
        args = {}
        args['indif_rho'] = rho
        args['combined_marginal_rho'] = rho1 # don't used in this phase, just as a penalty term
        args['dataset_name'] = 'temp_data'
        args['is_cal_depend'] = True
        args['marg_sel_threshold'] = 20000

        domain_list = [v for v in domain.values()]
        domain = Domain(df.columns, domain_list)
        dataset = Dataset(df, domain)
        # select_args['combined_marginal_rho'] = self.combined_marginal_rho
        # select_args['threshold'] = 5000
        
        marginals = marginal_selection(dataset, args)

        return marginals


        ########################################### generating process ###########################################
    def syn(self, n_sample, preprocesser, parent_dir, **kwargs):
        self.synthesize_records(n_sample)
        self.postprocessing(preprocesser, parent_dir)


        #######################################################################################################


        ########################################### helper function ###########################################
    def load_data(self):
        self.logger.info("loading dataset %s" % (self.dataset_name,))
        self.original_dataset = self.data_store.load_processed_data() 
    
    def load_data_from_df(self, df, domain):
        self.logger.info("loading dataset %s" % (self.dataset_name,))
        domain_list = [v for v in domain.values()]
        domain = Domain(df.columns, domain_list)
        self.original_dataset = Dataset(df, domain)

    def privacy_budget_allocation(self):
        # def _calculate_rho(epsilon):
        #     composition = AdvancedComposition()
        #     sigma = composition.gauss_zcdp(epsilon, self.delta, self.args['marg_add_sensitivity'], 1)
        
        #     return (self.args['marg_add_sensitivity'] ** 2 / (2.0 * sigma ** 2))
        
        # def _calculate_sigma(epsilon, num_margs):
        #     composition = AdvancedComposition()
    
        #     return composition.gauss_zcdp(epsilon, self.delta, self.args['marg_add_sensitivity'], num_margs)
        
        # self.total_rho = _calculate_rho(self.epsilon)
        self.indif_rho = self.total_rho * 0.1
        self.one_way_marginal_rho = self.total_rho * 0.1
        self.combined_marginal_rho = self.total_rho * 0.8

        self.logger.info('privacy budget allocation: marginal selection %s | 1 way marginals %s| combined marginals %s' % (self.indif_rho, self.one_way_marginal_rho, self.combined_marginal_rho))
    
    def obtain_marginal_list(self, dataset):
        '''
        
        implements algorithm 1(marginal selection) and algorithm 2(marginal combine) in paper
        
        '''
        if self.args['is_cal_marginals']:
            self.logger.info("selecting marginals")
    
            select_args = copy.deepcopy(self.args)
            # select_args['total_epsilon'] = self.epsilon 
            select_args['indif_rho'] = self.indif_rho
            select_args['combined_marginal_rho'] = self.combined_marginal_rho
            select_args['threshold'] = 5000
            
            marginals = marginal_selection(dataset, select_args) #alg1
            marginals = marginal_combine(dataset, select_args, marginals) #alg2
            self.data_store.save_marginal(marginals)
        else:
            marginals = self.data_store.load_marginal()
        
        return marginals

    def filter(self):
        self.logger.info("filtering attrs")
    
        filtered_dataset = Filter(self.original_dataset)
        if not self.no_filter:
            filtered_dataset.recode(self.one_way_marg_dict, self.gauss_sigma_4_one_way)
    
        return filtered_dataset
    
    def construct_marg(self, dataset, marginal):
        num_keys = reduce(lambda x, y: x * y, [dataset.domain.config[m] for m in marginal])
        self.logger.info("constructing %s margs, num_keys: %s" % (marginal, num_keys))

        marg = Marginal(dataset.domain.project(marginal), dataset.domain)
        marg.count_records(dataset.df.values)
        
        return marg
        
    def construct_margs(self, mode):
        if mode == 'one_way':
            self.logger.info("constructing one-way marginals")
            for attr in self.original_dataset.domain.attrs:
                self.one_way_marg_dict[(attr,)] = self.construct_marg(self.original_dataset, (attr,))
                self.singleton_key.append((attr,))
        
        elif mode == 'combined':
            self.logger.info("constructing combined marginals")
            for i, marginal in enumerate(self.sel_marg_name):
                self.logger.debug('%s th marginal' % (i,))
                self.combined_marg_dict[marginal] = self.construct_marg(self.filtered_dataset.dataset_recode, marginal)
                
    def anonymize_marg(self, marg, rho=0.0):
        sigma = math.sqrt(self.args['marg_add_sensitivity'] ** 2 / (2.0 * rho))
        noise = np.random.normal(scale=sigma, size=marg.num_key)
        marg.count += noise

        return marg

    def anonymize_margs(self, mode):
        divider = 0.0
        
        if mode == 'one_way':
            self.logger.info("anonymizing 1 way marginals")
            rho = self.one_way_marginal_rho / len(self.one_way_marg_dict)
            self.gauss_sigma_4_one_way = math.sqrt(self.args['marg_add_sensitivity'] ** 2 / (2.0 * rho))
            
            #Equal Gaussian for one way marginals
            for key, marg in self.one_way_marg_dict.items():
                self.anonymize_marg(marg, rho)
                
        elif mode == 'combined':
            self.logger.info("anonymizing combined marginals")
            rho = self.combined_marginal_rho
            
            #Weighted Gaussian for combined marginals
            for key, marg in self.combined_marg_dict.items():
                divider += math.pow(marg.num_key, 2.0 / 3.0)
            for key, marg in self.combined_marg_dict.items():
                marg.rho = rho * math.pow(marg.num_key, 2.0 / 3.0) / divider
                self.anonymize_marg(marg, rho=marg.rho)
        
        # update the marginal dict
        self.marg_dict = {**self.one_way_marg_dict, **self.combined_marg_dict}
                
    def consist_marginals(self, recode_domain, marg_dict):
        self.logger.info("consisting margs")
        
        consist_parameters = {
            "consist_iterations": self.args['consist_iterations'],
            "non_negativity": self.args['non_negativity'],
        }
        
        consistenter = Consistenter(marg_dict, recode_domain, consist_parameters)
        consistenter.consist_marginals()
        
        self.logger.info("consisted margs")

    def improving_convergence(self):
        logger = logging.getLogger("improving convergence")
        
        iterate_marginals, self.clip_layers = clip_graph(logger, self.filtered_dataset.dataset.domain, self.sel_marg_name, enable=self.args['append'])
        
        self.logger.info("iterate_marginals after clip_graph is %s" % (iterate_marginals,))
        
        self.iterate_keys = sep_graph(logger,self.original_dataset.domain, self.sel_marg_name, iterate_marginals, enable=self.args['sep_syn'])


    def synthesize_records(self, n_sample):
        self.args['num_synthesize_records'] = n_sample
        
        self.synthesized_df = pd.DataFrame(data=np.zeros([self.args['num_synthesize_records'], self.num_attributes], dtype=np.uint32),
                                           columns=self.original_dataset.domain.attrs)
        self.error_tracker = pd.DataFrame()
        
        # main procedure for synthesizing records
        for key, value in self.iterate_keys.items():
            self.logger.info("synthesizing for %s" % (key,))

            synthesizer = self._update_records(value)
            self.synthesized_df.loc[:, key] = synthesizer.update.df.loc[:, key]

            #ZL: error because of old append is deprecated
            #self.error_tracker = self.error_tracker.append(synthesizer.update.error_tracker)
            self.error_tracker = pd.concat([self.error_tracker, synthesizer.update.error_tracker])

    def _update_records(self, margs_iterate_key):
        update_config = {
            "alpha": self.args['update_rate_initial'],
            "alpha_update_method": self.args['update_rate_method'],
            "update_method": self.args['update_method'],
            "threshold": 0.0
        }

        singletons = {singleton: self.marg_dict[(singleton,)] for singleton in self.original_dataset.domain.attrs}

        synthesizer = UpdateConfig(self.filtered_dataset.dataset_recode.domain, self.args['num_synthesize_records'], update_config)
        synthesizer.update.initialize_records(margs_iterate_key, method=self.args['initialize_method'], singletons=singletons)

        for update_iteration in range(self.args['update_iterations']):
            self.logger.info("update round: %d" % (update_iteration,))

            synthesizer.update_alpha(update_iteration)
            margs_iterate_key = synthesizer.update_order(update_iteration, self.marg_dict, margs_iterate_key)

            for index, key in enumerate(margs_iterate_key):
                self.logger.info("updating %s marg: %s, num_key: %s" % (index, key, self.marg_dict[key].num_key))

                synthesizer.update_records(self.marg_dict[key], key, update_iteration)

        return synthesizer

    def postprocessing(self, preprocesser, save_path = None):
        '''
        
        complete the work of filter() and improving_convergence()
        
        '''
        logger = logging.getLogger("postprocessing dataset")
        
        append_attrs(logger, self.filtered_dataset.dataset.domain, self.clip_layers, self.synthesized_df, self.marg_dict)
        # decode records
        if not self.no_filter:
            self.filtered_dataset.decode(self.synthesized_df)

        preprocesser.reverse_data(self.synthesized_df, save_path)

        # self.synthesized_dataset = Dataset(self.synthesized_df, self.original_dataset.domain)
        # self.end_time = datetime.datetime.now()
        
        # if save_path is None: 
        #     logger = logging.getLogger("excution completed")
        #     logger.info("total excution time: %s" % (self.end_time - self.start_time))

        # self.data_store.save_synthesized_records(self.synthesized_dataset, save_path)
        
        ############################################################################################################