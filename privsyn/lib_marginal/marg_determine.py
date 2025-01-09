#####################################################################
#                                                                   #
#              This file implements the algorithm 1,2               #
#                             in paper                              #
#                                                                   #
#####################################################################

import logging
import itertools
import copy
import pickle
import math
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from privsyn.lib_marginal.marg import Marginal
from privsyn.lib_marginal.marg_select_helper import *
import privsyn.config as config

def marginal_selection(dataset, select_args):
    '''
        
    algorithm 1 in paper
        
    '''
    logger = logging.getLogger('choosing pairs')
        
    df = copy.deepcopy(dataset.df)
    dataset_domain = copy.deepcopy(dataset.domain)
    dataset_max_values = copy.deepcopy(dataset.domain)
    dataset_name = select_args['dataset_name']
    
    #obtaining indif(if needed)
    if select_args['is_cal_depend'] is True:

        df = transform_records_distinct_value(logger, df, dataset_domain)

        calculate_indif(logger, df, dataset_domain, dataset_max_values, dataset_name, select_args['indif_rho'])
    indif_df = pickle.load(open(config.DEPENDENCY_PATH + dataset_name, "rb"))
    
    #################################### main procedure of algorithm 1 #########################################
    gap = 1e10
    marginals = []
    selected_attrs = set()

    error = indif_df["error"].to_numpy() * df.shape[0]
    # error = self.dependency_df["error"].to_numpy()
    # ZL: np.float deprecated
    # num_cells = self.dependency_df["num_cells"].to_numpy().astype(np.float)
    num_cells = indif_df["num_cells"].to_numpy().astype(np.float64)
    overall_error = np.sum(error)
    selected = set()
    unselected = set(indif_df.index)

    # gauss_error_normalizer = np.sum(error) / (np.sum(np.power(num_cells, 0.75)) * np.sqrt(np.sum(np.sqrt(num_cells))))
    # gauss_error_normalizer = 1.0 / (np.sum(error) * self.df.shape[0])
    gauss_error_normalizer = 1.0

    while gap > select_args['marg_sel_threshold']:
        error_new = np.sum(error)
        selected_index = None

        for j in unselected:
            select_candidate = selected.union({j})

            cells_square_sum = np.sum(
                np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
            gauss_constant = np.sqrt(cells_square_sum / (math.pi * select_args['combined_marginal_rho']))
            gauss_error = np.sum(
                gauss_constant * np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
            
            # if select_args['noise_add_method'] == "A3":
            #     cells_square_sum = np.sum(
            #         np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
            #     gauss_constant = np.sqrt(cells_square_sum / (math.pi * select_args['combined_marginal_rho']))
            #     gauss_error = np.sum(
            #         gauss_constant * np.power(num_cells[list(select_candidate)], 2.0 / 3.0))
            # elif select_args['noise_add_method'] == "A1" or select_args['noise_add_method'] == "A2":
            #     gauss_constant = np.sqrt(
            #         (select_args['marg_add_sensitivity'] ** 2 * len(select_candidate)) / (2.0 * select_args['combined_marginal_rho']))
            #     gauss_error = np.sum(
            #         gauss_constant * num_cells[list(select_candidate)])
            # else:
            #     raise Exception("invalid noise add method")

            gauss_error *= gauss_error_normalizer

            pairwise_error = np.sum(
                error[list(unselected.difference(select_candidate))])
            error_temp = gauss_error + pairwise_error

            if error_temp < error_new:
                selected_index = j
                error_new = error_temp

        gap = overall_error - error_new
        overall_error = error_new
        selected.add(selected_index)
        unselected.remove(selected_index)

        first_attr, second_attr = indif_df.loc[selected_index, "first_attr"], indif_df.loc[
            selected_index, "second_attr"]
        marginals.append((first_attr, second_attr))
        selected_attrs.update((first_attr, second_attr))

        logger.info("select %s marginal: %s | gap: %s" %(len(marginals), (first_attr, second_attr), gap))
    ##########################################################################################################
        
    #handle_isolated_attrs
    marginals = handle_isolated_attrs(dataset_domain, selected_attrs, indif_df, marginals, method="connect", sort=True)

    logger.info('marginals after selection: %s' % (marginals, ))
    
    return marginals #这里返回的是attr


def marginal_combine(dataset, select_args, marginals):
    '''
        
    algorithm 2 in paper
        
    '''
    if not select_args['is_combine']:
        return marginals
    
    logger = logging.getLogger('combining pairs')
    logger.info("%s marginals before combining" % (len(marginals)))
        
    #constructing graph
    graph = nx.Graph()
    graph.add_nodes_from(dataset.domain.attrs)

    for m in marginals:
        graph.add_edges_from(itertools.combinations(m, 2))

    #identifying cliques
    all_cliques = nx.enumerate_all_cliques(graph)
    size_cliques = defaultdict(list)

    for clique in all_cliques:
        size_cliques[len(clique)].append(clique)
        
    #combining marginals
    combined_marginals = []
    selected_attrs = set()

    for size in range(len(size_cliques), 2, -1):
        for clique in size_cliques[size]:
            if len(set(clique) & selected_attrs) <= 2 and dataset.domain.size(clique) < select_args['threshold']:
                combined_marginals.append(tuple(clique))
                selected_attrs.update(clique)
                    
    #identifying missing depend
    missing_depend = copy.deepcopy(marginals)

    for marginal in combined_marginals:
        for marg in itertools.combinations(marginal, 2):
            if marg in missing_depend:
                missing_depend.remove(marg)
            elif (marg[1], marg[0]) in missing_depend:
                missing_depend.remove((marg[1], marg[0]))

    marginals = combined_marginals + missing_depend

    logger.info("%s marginals after combining" % (len(marginals)))
        
    return marginals
