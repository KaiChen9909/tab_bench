#########################################################################################################
#                                                                                                       #
#                       The filter class, including recode and decode functions.                        #
#         Note that we need to create a new value while filtering. Thus recoding is necessary.          #
#                                                                                                       #
#########################################################################################################

import logging
import copy

import numpy as np

from privsyn.lib_marginal.marg import Marginal
from privsyn.lib_dataset.dataset import Dataset


class Filter:
    def __init__(self, dataset):
        self.logger = logging.getLogger("filter")

        # if no filter is conducted then dataset_recode is dataset
        self.dataset = dataset
        self.dataset_recode = dataset 

        self.significant_cell_indices = {}
        self.group_cell_indices = {}

    def recode(self, one_way_marg_dict, gauss_sigma):
        '''
        
        In this function, significant cell values and group cell values are selected and stored.
        At the same time, the attribute value is replaced with the index of the unique value on that attribute.
        Low-count values are filterd and combined.
        
        '''
        dataset_recode = Dataset(copy.deepcopy(self.dataset.df), copy.deepcopy(self.dataset.domain))

        for index, attr_name in enumerate(self.dataset.domain.attrs):
            self.logger.info("recoding %s attr %s" % (index, attr_name))

            marg = one_way_marg_dict[(attr_name, )]
            marg.non_negativity("N1")

            num_records = self.dataset.df.values.shape[0]
            attr_index = self.dataset.domain.attr_index_mapping[attr_name]
            # records = self.dataset.df.to_numpy()
            record = self.dataset.df.values[:, attr_index]

            # find the significant value
            significant_cell_indices = np.where(marg.count >= 3.0 * gauss_sigma)[0]
            group_cell_indices = np.where(marg.count < 3.0 * gauss_sigma)[0]

            if group_cell_indices.size != 0:
                # encode the cells with values above threshold
                significant_records_indices = np.where(np.isin(record, significant_cell_indices))[0]

                # update records
                significant_records = record[significant_records_indices]
                num_repeat = np.zeros(significant_cell_indices.size, dtype=np.uint32)
                unique_value, count = np.unique(significant_records, return_counts=True)
                num_repeat[np.isin(significant_cell_indices, unique_value)] = count 
                # np.isin() is necessary because some values in marg don't actually exists in raw dataset. 
                # (marg has been added noise, producing unexistent value count)
        
                sort_indices = np.argsort(significant_records)
                significant_records[sort_indices] = np.repeat(np.arange(significant_cell_indices.size), num_repeat)
                
                new_record = np.zeros(num_records, dtype=np.uint32)
                new_record[significant_records_indices] = significant_records

                # encode the cells with values below threshold
                remain_indices = np.setdiff1d(np.arange(num_records), significant_records_indices)
                new_record[remain_indices] = len(significant_cell_indices)# a new value is created

                # update dataset
                dataset_recode.change_column(attr_name, new_record, len(significant_cell_indices) + 1)
                
                # update one-way marginals
                low_count_sum = marg.count[group_cell_indices].sum()
                marg.count = marg.count[np.isin(np.arange(marg.num_key), significant_cell_indices)]
                marg.count= np.append(marg.count, low_count_sum)

            self.significant_cell_indices[attr_name] = significant_cell_indices
            self.group_cell_indices[attr_name] = group_cell_indices

            self.logger.info("remain %s values" % (dataset_recode.domain.shape[attr_index]))
            
        self.dataset_recode = dataset_recode
        
        #doing this to update other parameters in one-way marginals
        #won't consume privacy budget cuz it isn't calculating marginals
        for index, attr_name in enumerate(self.dataset.domain.attrs):
            marg = one_way_marg_dict[(attr_name, )]
            tmp_marg = Marginal(dataset_recode.domain.project((attr_name, )), dataset_recode.domain)
            tmp_marg.count = marg.count
            one_way_marg_dict[(attr_name, )] = tmp_marg

    def decode(self, df):
        for attr_name in self.dataset.domain.attrs:
            self.logger.info("decoding attribute %s" % (attr_name,))

            significant_cell_indices = self.significant_cell_indices[attr_name]
            group_cell_indices = self.group_cell_indices[attr_name]
            encode_record = np.copy(df[attr_name])
            decode_record = np.zeros(encode_record.size, dtype=np.uint32)

            # decode the significant value
            for anchor_value in range(significant_cell_indices.size):
                anchor_value_indices = np.where(encode_record == anchor_value)[0]
                decode_record[anchor_value_indices] = significant_cell_indices[anchor_value]

            # decode the grouped value
            if group_cell_indices.size != 0:
                anchor_value_indices = np.where(encode_record == significant_cell_indices.size)[0]

                if anchor_value_indices.size != 0:
                    group_value_dist = np.full(group_cell_indices.size, 1.0 / group_cell_indices.size)
                    group_value_cumsum = np.cumsum(group_value_dist)
                    start = 0

                    for index, value in enumerate(group_value_cumsum):
                        end = int(round(value * anchor_value_indices.size))

                        decode_record[anchor_value_indices[start: end]] = group_cell_indices[index]

                        start = end

            df[attr_name] = decode_record
