import logging
import copy

import numpy as np

from privsyn.lib_marginal.marg import Marginal


class Consistenter:
    class SubsetWithDependency:
        def __init__(self, attr_set):
            # a list of categories
            self.attr_set = attr_set
            # a set of tuples this object depends on
            self.dependency = set()
    
    def __init__(self, margs, dataset_domain, consist_parameters):
        self.logger = logging.getLogger("Consistenter")
        
        self.margs = margs
        self.dataset_domain = dataset_domain
        self.num_categories = np.array(dataset_domain.shape)
        self.iterations = consist_parameters["consist_iterations"]
        self.non_negativity = consist_parameters["non_negativity"]

    def _compute_dependency(self):
        subsets_with_dependency = {}

        for marg_key in self.margs:
            # create a dependency subset for each marg
            new_subset = self.SubsetWithDependency(set(marg_key))
            subsets_temp = copy.deepcopy(subsets_with_dependency)

            for subset_key, subset_value in subsets_temp.items():
                # sort here to avoid producing multiple keys with the same set
                attr_intersection = sorted(subset_value.attr_set & set(marg_key))

                if attr_intersection:
                    # add interacted attrs as dependency subset
                    if tuple(attr_intersection) not in subsets_with_dependency:
                        intersection_subset = self.SubsetWithDependency(set(attr_intersection))
                        subsets_with_dependency[tuple(attr_intersection)] = intersection_subset

                    # add dependency to subset_key, and avoid regarding self as dependency
                    if not set(attr_intersection) == set(subset_key):
                        subsets_with_dependency[subset_key].dependency.add(tuple(attr_intersection))

                    # add dependency to the new marg

                    if not set(attr_intersection) == set(marg_key):
                        new_subset.dependency.add(tuple(attr_intersection))

                    # add dependency to other subsets
                    for sub_key, sub_value in subsets_with_dependency.items():
                        if set(attr_intersection) < sub_value.attr_set:
                            subsets_with_dependency[sub_key].dependency.add(tuple(attr_intersection))

            subsets_with_dependency[marg_key] = new_subset

        return subsets_with_dependency
    
    def consist_marginals(self):
        def find_subset_without_dependency():
            for key, subset in subsets_with_dependency_temp.items():
                if not subset.dependency:
                    return key, subset
            
            return None, None
        
        def find_margs_containing_target(target):
            result = []
            
            for _, marg in self.margs.items():
                if target <= marg.attr_set:
                    result.append(marg)
            
            return result

        def remove_subset_from_dependency(target):
            for _, subset in subsets_with_dependency_temp.items():
                if target in subset.dependency:
                    subset.dependency.remove(target)
        
        def consist_on_subset(target, target_margs):
            common_marg = Marginal(self.dataset_domain.project(target), self.dataset_domain)
            common_marg.init_consist_parameters(len(target_margs))

            for index, marg in enumerate(target_margs):
                common_marg.project_from_bigger_marg(marg, index)

            common_marg.calculate_delta()

            for index, marg in enumerate(target_margs):
                marg.update_marg(common_marg, index)

        # calculate necessary variables
        for key, marg in self.margs.items():
            assert np.array_equal(marg.num_categories, self.num_categories)
            
            marg.calculate_tuple_key()
            # marg.generate_attributes_index_set()
            # marg.get_sum()

        # calculate the dependency relationship
        subsets_with_dependency = self._compute_dependency()
        self.logger.debug("dependency computed")
        
        # ripple steps needs several iterations
        # for i in range(self.iterations):
        non_negativity = True
        iterations = 0

        while non_negativity and iterations < self.iterations:
            self.logger.info("consist for round %s" % (iterations,))

            # first make sure summation are the same
            consist_on_subset(set(), [marg for _, marg in self.margs.items()])
            subsets_with_dependency_temp = copy.deepcopy(subsets_with_dependency)
            
            # consist margs in the dependency tree
            while len(subsets_with_dependency_temp) > 0:
                key, subset = find_subset_without_dependency()
                target_margs = find_margs_containing_target(subset.attr_set)

                # only if the number of target margs larger than 1, one need to consist margs
                if len(target_margs) > 1:
                    consist_on_subset(subset.attr_set, target_margs)
                    remove_subset_from_dependency(key)

                subsets_with_dependency_temp.pop(key, None)

            self.logger.debug("consist finish")

            # if iterations % 10 == 0:
            #     self.check_whole_consistency()

            # ensure all cells in all margs are non-negative
            margs_count = 0
            
            for key, marg in self.margs.items():
                if (marg.count < 0.0).any():
                    marg.non_negativity(self.non_negativity, iterations)
                    marg.get_sum()
                else:
                    margs_count += 1
                
                if margs_count == len(self.margs):
                    self.logger.info("finish in %s round" % (iterations,))
                    non_negativity = False

            self.logger.debug("non-negativity finish")
            
            iterations += 1

        # calculate normalized count
        for key, marg in self.margs.items():
            marg.calculate_normalize_count()
            # marg.get_sum()
