#####################################################################
#                                                                   #
#           function marginal_selection's hepler function           #
#                                                                   #
#####################################################################

import logging
import pickle
import copy
import math

import numpy as np
import pandas as pd

from method.privsyn.lib_marginal.marg import Marginal
import method.privsyn.config as config

def transform_records_distinct_value(logger, df, dataset_domain):
    '''

    Replaces the attribute value with the index of the unique value on that attribute.
    Fix the domain size at the same time.
    
    It's vital for marginal caculation to function normally.

    '''
    logger.info("transforming records")

    distinct_shape = []
    for attr_index, attr in enumerate(dataset_domain.attrs):
        record = np.copy(df.loc[:, attr])
        unique_value = np.unique(record)
        distinct_shape.append(unique_value.size)

        for index, value in enumerate(unique_value):
            indices = np.where(record == value)[0]
            # self.df.loc[indices, attr] = index
            # df.value[indices, attr_index] = index
            df.iloc[indices, attr_index] = index
    dataset_domain.shape = tuple(distinct_shape)
    dataset_domain.config = dict(
        zip(dataset_domain.attrs, distinct_shape))
    logger.info("transformed records")

    return df


def calculate_indif(logger, df, dataset_domain, original_domain, dataset_name, rho):
    logger.info("calculating pair indif")

    indif_df = pd.DataFrame(
        columns=["first_attr", "second_attr", "num_cells", "error"])
    indif_index = 0

    for first_index, first_attr in enumerate(dataset_domain.attrs[:-1]): #2-ways margin到倒数第二个即可
        first_marg = Marginal(dataset_domain.project(
            first_attr), dataset_domain) # first one-way marginal
        first_marg.count_records(df.values)
        first_histogram = first_marg.calculate_normalize_count()

        for second_attr in dataset_domain.attrs[first_index + 1:]:
            logger.info("calculating [%s, %s]" %
                             (first_attr, second_attr))

            second_marg = Marginal(dataset_domain.project(
                second_attr), dataset_domain) # second one-way marginal
            second_marg.count_records(df.values)
            second_histogram = second_marg.calculate_normalize_count()

            # calculate real 2-way marginal
            pair_marg = Marginal(dataset_domain.project(
                (first_attr, second_attr)), dataset_domain)
            pair_marg.count_records(df.values)
            pair_marg.calculate_count_matrix()
            # pair_distribution = pair_marg.calculate_normalize_count()

            # calculate 2-way marginal assuming independent
            independent_pair_distribution = np.outer(
                first_histogram, second_histogram)

            # calculate the errors
            normalize_pair_marg_count = pair_marg.count_matrix / \
                np.sum(pair_marg.count_matrix)
            error = np.sum(np.absolute(
                normalize_pair_marg_count - independent_pair_distribution))

            num_cells = original_domain.config[first_attr] * \
                original_domain.config[second_attr]
            indif_df.loc[indif_index] = [
                first_attr, second_attr, num_cells, error]

            indif_index += 1

    # add noise
    n = np.sum(pair_marg.count_matrix)
    if rho != 0.0:
        indif_df.error *= n
        indif_df.error += np.random.normal(
            scale=np.sqrt(8 * indif_df.shape[0]/rho), size=indif_df.shape[0])

    # publish indif
    pickle.dump(indif_df, open(
        config.DEPENDENCY_PATH + dataset_name, "wb"))

    logger.info("calculated pair indif")

def handle_isolated_attrs(dataset_domain, selected_attrs, indif_df, marginals, method="isolate", sort=False):
    # find attrs that does not appear in any of the pairwise marginals
    missing_attrs = set(dataset_domain.attrs) - selected_attrs

    if sort:
        # self.dependency_df["error"] /= np.sqrt(self.dependency_df["num_cells"].astype("float"))
        indif_df.sort_values(
            by="error", ascending=False, inplace=True)
        indif_df.reset_index(drop=True, inplace=True)

    for attr in missing_attrs:
        if method == "isolate":
            marginals.append((attr,))

        elif method == "connect":
            match_missing_df = indif_df.loc[
                (indif_df["first_attr"] == attr) | (indif_df["second_attr"] == attr)]
            match_df = match_missing_df.loc[(match_missing_df["first_attr"].isin(selected_attrs)) | (
                match_missing_df["second_attr"].isin(selected_attrs))]
            match_df.reset_index(drop=True, inplace=True)
            marginals.append(
                (match_df.loc[0, "first_attr"], match_df.loc[0, "second_attr"]))

    return marginals
