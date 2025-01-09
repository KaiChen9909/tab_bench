import numpy as np

class Marginal:
    def __init__(self, marg_domain, dataset_domain):

        # Indicator is a list with the length of total attribute numbers. Each element takes 0 or 1 as its value.
        # Value 1 represents that the corresponding attribute needs to be preserved.
        self.indicator = np.zeros(len(dataset_domain.attrs), dtype=np.uint8)
        self.indicator[[dataset_domain.attr_index_mapping[attr]
                        for attr in marg_domain.attrs]] = 1
        self.num_categories = np.array(dataset_domain.shape)

        self.attributes_index = np.nonzero(self.indicator)[0] #所有margin的attr的index
        self.attr_set = set(marg_domain.attrs)

        # self.num_key represents the number of different possible combinations in the marg
        self.num_key = np.product(self.num_categories[self.attributes_index])
        self.num_attributes = self.indicator.shape[0]
        self.ways = np.count_nonzero(self.indicator)

        self.encode_num = np.zeros(self.ways, dtype=np.uint32)
        self.cum_mul = np.zeros(self.ways, dtype=np.uint32)

        self.count = np.zeros(self.num_key)
        self.rho = 0.0

        self.calculate_encode_num(self.num_categories)

    ########################################### general functions ####################################
    def calculate_encode_num(self, num_categories): #将多维的组合[a,b,c]降到一维，用于计算histogram
        if self.ways != 0:
            categories_index = self.attributes_index

            categories_num = num_categories[categories_index]
            categories_num = np.roll(categories_num, 1)
            categories_num[0] = 1
            self.cum_mul = np.cumprod(categories_num)

            categories_num = num_categories[categories_index]
            categories_num = np.roll(categories_num, self.ways - 1)
            categories_num[-1] = 1
            categories_num = np.flip(categories_num)
            self.encode_num = np.flip(np.cumprod(categories_num))

    def calculate_tuple_key(self):
        self.tuple_key = np.zeros([self.num_key, self.ways], dtype=np.uint32)

        if self.ways != 0:
            for i in range(self.attributes_index.shape[0]):
                index = self.attributes_index[i]
                categories = np.arange(self.num_categories[index])
                column_key = np.tile(
                    np.repeat(categories, self.encode_num[i]), self.cum_mul[i])

                self.tuple_key[:, i] = column_key
        else:
            self.tuple_key = np.array([0], dtype=np.uint32)
            self.num_key = 1

    def count_records(self, records):
        encode_key,count_num = np.unique(records[:, self.attributes_index], return_counts=True)

        encode_records = np.matmul(
            records[:, self.attributes_index], self.encode_num)
        # encode_records = records[:, self.attributes_index]

        encode_key, count_num = np.unique(encode_records, return_counts=True)
        indices = np.where(np.isin(np.arange(self.num_key), encode_key))[0]
    
        #try:
        self.count[indices] = count_num
        #except:
        #    raise Exception(
        #        "num_key is smaller than the largest value, check domain shape")

    def calculate_normalize_count(self):
        self.normalize_count = self.count / np.sum(self.count)

        return self.normalize_count

    def calculate_count_matrix(self):
        shape = []

        for attri in self.attributes_index:
            shape.append(self.num_categories[attri])

        self.count_matrix = np.copy(self.count).reshape(tuple(shape))

        return self.count_matrix

    def calculate_count_matrix_2d(self, row_attr_index):
        attr_index = np.where(self.attributes_index == row_attr_index)[0]
        shape = [1, 1]

        for attri in self.attributes_index:
            if attri == row_attr_index:
                shape[0] *= self.num_categories[attri]
            else:
                shape[1] *= self.num_categories[attri]

        self.count_matrix_2d = np.zeros(shape)

        for value in range(shape[0]):
            indices = np.where(self.tuple_key[:, attr_index] == value)[0]
            self.count_matrix_2d[value] = self.count[indices]

        return self.count_matrix_2d

    def reserve_original_count(self):
        self.original_count = self.count

    def get_sum(self):
        self.sum = np.sum(self.count)

    def generate_attributes_index_set(self):
        self.attributes_set = set(self.attributes_index)

    ################################### functions for outside invoke #########################
    def calculate_encode_num_general(self, attributes_index):
        categories_index = attributes_index

        categories_num = self.num_categories[categories_index]
        categories_num = np.roll(categories_num, attributes_index.size - 1)
        categories_num[-1] = 1
        categories_num = np.flip(categories_num)
        encode_num = np.flip(np.cumprod(categories_num))

        return encode_num

    def count_records_general(self, records):
        count = np.zeros(self.num_key)

        encode_records = np.matmul(
            records[:, self.attributes_index], self.encode_num)
        encode_key, value_count = np.unique(encode_records, return_counts=True)

        indices = np.where(np.isin(np.arange(self.num_key), encode_key))[0]
        count[indices] = value_count

        return count

    def calculate_normalize_count_general(self, count):
        return count / np.sum(count)

    def calculate_count_matrix_general(self, count):
        shape = []

        for attri in self.attributes_index:
            shape.append(self.num_categories[attri])

        return np.copy(count).reshape(tuple(shape))

    def calculate_tuple_key_general(self, unique_value_list):
        self.tuple_key = np.zeros([self.num_key, self.ways], dtype=np.uint32)

        if self.ways != 0:
            for i in range(self.attributes_index.shape[0]):
                categories = unique_value_list[i]
                column_key = np.tile(
                    np.repeat(categories, self.encode_num[i]), self.cum_mul[i])

                self.tuple_key[:, i] = column_key
        else:
            self.tuple_key = np.array([0], dtype=np.uint32)
            self.num_key = 1

    def project_from_bigger_marg_general(self, bigger_marg):
        encode_num = np.zeros(self.num_attributes, dtype=np.uint32)
        encode_num[self.attributes_index] = self.encode_num
        encode_num = encode_num[bigger_marg.attributes_index]

        encode_tuple_key = np.matmul(bigger_marg.tuple_key, encode_num)
        self.count = np.bincount(
            encode_tuple_key, weights=bigger_marg.count, minlength=self.num_key)

        # for i in range(self.num_key):
        #     key_index = np.where(encode_tuple_key == i)[0]
        #     self.count[i] = np.sum(bigger_marg.count[key_index])

    ######################################## functions for consistency #######################
    ############ used in commom marg #############
    def init_consist_parameters(self, num_target_margs):
        self.summations = np.zeros([self.num_key, num_target_margs])
        self.weights = np.zeros(num_target_margs)
        self.rhos = np.zeros(num_target_margs)

    def calculate_delta(self):
        weights = self.rhos * self.weights
        target = np.matmul(self.summations, weights) / np.sum(weights)
        self.delta = - (self.summations.T - target).T * weights

    def project_from_bigger_marg(self, bigger_marg, index):
        encode_num = np.zeros(self.num_attributes, dtype=np.uint32)
        encode_num[self.attributes_index] = self.encode_num
        encode_num = encode_num[bigger_marg.attributes_index]

        encode_tuple_key = np.matmul(bigger_marg.tuple_key, encode_num)

        self.weights[index] = 1.0 / np.product(
            self.num_categories[np.setdiff1d(bigger_marg.attributes_index, self.attributes_index)])
        self.rhos[index] = bigger_marg.rho

        self.summations[:, index] = np.bincount(
            encode_tuple_key, weights=bigger_marg.count, minlength=self.num_key)

        # for i in range(self.num_key):
        #     key_index = np.where(encode_tuple_key == i)[0]
        #     self.summations[i, index] = np.sum(bigger_marg.count[key_index])

    ############### used in margs to be consisted ###############
    def update_marg(self, common_marg, index):
        encode_num = np.zeros(self.num_attributes, dtype=np.uint32)
        encode_num[common_marg.attributes_index] = common_marg.encode_num
        encode_num = encode_num[self.attributes_index]

        encode_tuple_key = np.matmul(self.tuple_key, encode_num)

        sort_indices = np.argsort(encode_tuple_key)
        _, count = np.unique(encode_tuple_key, return_counts=True)
        np.add.at(self.count, sort_indices, np.repeat(
            common_marg.delta[:, index], count))

        # for i in range(common_marg.num_key):
        #     key_index = np.where(encode_tuple_key == i)[0]
        #     self.count[key_index] += common_marg.delta[i, index]

    ######################################### non-negative functions ####################################
    def non_negativity(self, method, iteration=-1):
        if method == "N3":
            assert iteration != -1

        if method == "N1":
            self.count = norm_cut(self.count)
        elif method == "N2":
            self.count = norm_sub(self.count)
        elif method == "N3":
            if iteration < 400:
                self.count = norm_sub(self.count)
            else:
                self.count = norm_cut(self.count)
        else:
            raise Exception("non_negativity method is invalid")


def norm_sub(count):
    summation = np.sum(count)
    lower_bound = 0.0
    upper_bound = - np.sum(count[count < 0.0])
    current_summation = 0.0
    delta = 0.0

    while abs(summation - current_summation) > 1.0:
        delta = (lower_bound + upper_bound) / 2.0
        new_count = count - delta
        new_count[new_count < 0.0] = 0.0
        current_summation = np.sum(new_count)

        if current_summation < summation:
            upper_bound = delta
        elif current_summation > summation:
            lower_bound = delta
        else:
            break

    count = count - delta
    count[count < 0.0] = 0.0

    return count


def norm_cut(count):
    # set all negative value to 0.0
    negative_indices = np.where(count < 0.0)[0]
    negative_total = abs(np.sum(count[negative_indices]))
    count[negative_indices] = 0.0

    # find all positive value and sort them in ascending order
    positive_indices = np.where(count > 0.0)[0]

    if positive_indices.size != 0:
        positive_sort_indices = np.argsort(count[positive_indices])
        sort_cumsum = np.cumsum(count[positive_indices[positive_sort_indices]])

        # set the smallest positive value to 0.0 to preserve the total density
        threshold_indices = np.where(sort_cumsum <= negative_total)[0]

        if threshold_indices.size == 0:
            count[positive_indices[positive_sort_indices[0]]
                  ] = sort_cumsum[0] - negative_total
        else:
            count[positive_indices[positive_sort_indices[threshold_indices]]] = 0.0
            next_index = threshold_indices[-1] + 1

            if next_index < positive_sort_indices.size:
                count[positive_indices[positive_sort_indices[next_index]]] = sort_cumsum[
                    next_index] - negative_total
    else:
        count[:] = 0.0

    return count
