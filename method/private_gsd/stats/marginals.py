import itertools
import jax
import jax.numpy as jnp
import chex
from method.private_gsd.utils import Dataset, Domain
from method.private_gsd.stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np
from typing import List


class Marginals(AdaptiveStatisticState):

    def __init__(self, domain, kway_combinations, k, bins=(32,)):
        self.domain = domain
        self.kway_combinations = kway_combinations
        self.k = k
        self.bins = list(bins)
        self.workload_positions = []
        self.workload_sensitivity = []
        self.set_up_stats()

    def __str__(self):
        return f'Marginals'

    def get_num_workloads(self):
        return len(self.workload_positions)

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return self.workload_positions[workload_id]

    def is_workload_numeric(self, cols):
        for c in cols:
            if c in self.domain.get_numeric_cols():
                return True
        return False

    def set_up_stats(self):

        queries = []
        self.workload_positions = []
        for marginal in tqdm(self.kway_combinations, desc='Setting up Marginals.'):
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)
            bins = self.bins if self.is_workload_numeric(marginal) else [-1]
            start_pos = len(queries)
            for bin in bins:
                intervals = []
                for att in marginal:
                    size = self.domain.size(att)
                    if size > 1:
                        upper = np.linspace(0, size, num=size+1)[1:]
                        lower = np.linspace(0, size, num=size+1)[:-1]
                        # lower = lower.at[0].set(-0.01)
                        interval = list(np.vstack((upper, lower)).T - 0.1)
                        intervals.append(interval)
                    else:
                        upper = np.linspace(0, 1, num=bin+1)[1:]
                        lower = np.linspace(0, 1, num=bin+1)[:-1]
                        upper[-1] = 1.01
                        # upper = upper.at[-1].set(1.01)
                        interval = list(np.vstack((upper, lower)).T)
                        intervals.append(interval)
                for v in itertools.product(*intervals):
                    v_arr = np.array(v)
                    upper = v_arr.flatten()[::2]
                    lower = v_arr.flatten()[1::2]
                    q = np.concatenate((indices, upper, lower))
                    queries.append(q)  # (i1, i2), ((a1, a2), (b1, b2))
            end_pos = len(queries)
            self.workload_positions.append((start_pos, end_pos))
            self.workload_sensitivity.append(jnp.sqrt(2 * len(bins)))

        self.queries = jnp.array(queries)

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return self.workload_sensitivity[workload_id] / N

    def _get_dataset_statistics_fn(self, workload_ids=None, jitted: bool = False):
        if jitted:
            workload_fn = jax.jit(self._get_workload_fn(workload_ids))
        else:
            workload_fn = self._get_workload_fn(workload_ids)

        def data_fn(data: Dataset):
            X = data.to_numpy()
            return workload_fn(X)
        return data_fn

    def _get_workload_fn(self, workload_ids=None):
        # query_ids = []
        if workload_ids is None:
        #     these_queries = self.queries
            query_ids = jnp.arange(self.queries.shape[0])
        else:
            query_positions = []
            for stat_id in workload_ids:
                a, b = self.workload_positions[stat_id]
                q_pos = jnp.arange(a, b)
                query_positions.append(q_pos)
            query_ids = jnp.concatenate(query_positions)

        return self._get_stat_fn(query_ids)

    def _get_stat_fn(self, query_ids):
        def answer_fn(x_row: chex.Array, query_single: chex.Array):
            I = query_single[:self.k].astype(int)
            U = query_single[self.k:2 * self.k]
            L = query_single[2 * self.k:3 * self.k]
            t1 = (x_row[I] < U).astype(int)
            t2 = (x_row[I] >= L).astype(int)
            t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
            answers = jnp.prod(t3)
            return answers

        these_queries = self.queries[query_ids]
        temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))

        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, these_queries), None
        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]
        return stat_fn

    @staticmethod
    def get_kway_categorical(domain: Domain, k):
        kway_combinations = [list(idx) for idx in itertools.combinations(domain.get_categorical_cols(), k)]
        return Marginals(domain, kway_combinations, k, bins=[2])

    @staticmethod
    def get_all_kway_combinations(domain, k, bins=(32,), max_size=None):
        if max_size is  None:
            kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
        else:
            kway_combinations = []
            for idx in itertools.combinations(domain.attrs, k):
                total_size = 1
                for col in idx:
                    sz = domain.size(col)
                    if sz == 1:
                        sz = sum(bins)
                    total_size = total_size * sz
                if total_size <= max_size:
                    # print(f'Adding ', idx, ' size=', total_size)
                    kway_combinations.append(list(idx))
            # kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)
            #                      if domain.size(idx) <= max_size]
        return Marginals(domain, kway_combinations, k, bins=bins)


    @staticmethod
    def get_outside_kway_combinations(marginals, domain, k, bins):
        kway_combinations = [list(marginal) for marginal in marginals]
        return Marginals(domain, kway_combinations, k, bins=bins)


    @staticmethod
    def get_all_kway_mixed_combinations_v1(domain, k, bins=(32,)):
        num_numeric_feats = len(domain.get_numeric_cols())
        k_real = num_numeric_feats
        kway_combinations = []
        for cols in itertools.combinations(domain.attrs, k):
            count_disc = 0
            count_real = 0
            for c in list(cols):
                if c in domain.get_numeric_cols():
                    count_real += 1
                else:
                    count_disc += 1
            if count_disc > 0 and count_real > 0:
                kway_combinations.append(list(cols))

        return Marginals(domain, kway_combinations, k, bins=bins)


######################################################################
## TEST
######################################################################
