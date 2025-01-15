import itertools
import jax
import jax.numpy as jnp
from method.private_gsd.utils import Dataset
from method.private_gsd.utils.utils_data import Domain
from method.private_gsd.stats import AdaptiveStatisticState
import numpy as np
import chex
from tqdm import tqdm


class Halfspace(AdaptiveStatisticState):


    def __init__(self, domain: Domain,
                 k_cat: int,
                 cat_kway_combinations: list,
                 rng: chex.PRNGKey,
                 num_random_halfspaces: int,
                 ):
        """
        :param domain:
        :param kway_combinations:
        :param num_random_halfspaces: number of random halfspaces temp for each marginal that contains a real-valued feature
        """
        self.domain = domain
        self.num_hs_samples = num_random_halfspaces
        self.rng = rng

        self.cat_kway_combinations = cat_kway_combinations
        self.k = k_cat

        self.halfspace_keys = jax.random.split(self.rng, self.num_hs_samples)
        self.workload_positions = []
        self.workload_sensitivity = []

        self.set_up_stats()


    def __str__(self):
        return f'Halfspaces'

    @staticmethod
    def get_kway_random_halfspaces(domain: Domain,
                                   k: int,
                                   rng: chex.PRNGKey,
                                   random_hs: int = 500):
        kway_combinations = []
        for cols in itertools.combinations(domain.get_categorical_cols(), k):
            kway_combinations.append(list(cols))
        return Halfspace(domain, k_cat=k, cat_kway_combinations=kway_combinations, rng=rng,
                         num_random_halfspaces=random_hs)


    def get_num_workloads(self):
        return len(self.workload_positions)

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        return self.workload_positions[workload_id]

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        return self.workload_sensitivity[workload_id] / N

    def set_up_stats(self):

        queries = []
        self.workload_positions = []
        for marginal in tqdm(self.cat_kway_combinations, desc='Setting up half-spaces'):
            assert len(marginal) == self.k
            indices = self.domain.get_attribute_indices(marginal)

            for halfspace_pos in range(self.num_hs_samples):
                start_pos = len(queries)
                intervals = []
                for att in marginal:
                    size = self.domain.size(att)
                    assert size>1
                    upper = np.linspace(0, size, num=size+1)[1:]
                    lower = np.linspace(0, size, num=size+1)[:-1]
                    # lower = lower.at[0].set(-0.01)

                    intervals_arr = np.vstack((upper, lower)).T - 0.1
                    interval_list = list(intervals_arr)
                    intervals.append(interval_list)

                for v in itertools.product(*intervals):
                    v_arr = np.array(v)
                    upper = v_arr.flatten()[::2]
                    lower = v_arr.flatten()[1::2]
                    q = np.concatenate((indices, upper, lower, np.array([halfspace_pos])))
                    queries.append(q)
                end_pos = len(queries)
                self.workload_positions.append((start_pos, end_pos))
                self.workload_sensitivity.append(jnp.sqrt(2))

        self.queries = jnp.array(queries)

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
        """
        Returns marginals function and sensitivity
        :return:
        """
        dim = len(self.domain.attrs)
        numeric_cols = self.domain.get_numeric_cols()
        num_idx = self.domain.get_attribute_indices(numeric_cols).astype(int)
        numeric_dim = num_idx.shape[0]


        def answer_fn(x_row: chex.Array, query_single: chex.Array):
            I = query_single[:self.k].astype(int)
            U = query_single[self.k:2*self.k]
            L = query_single[2*self.k:3*self.k]
            key_id = query_single[-1].astype(int)
            halfspace_key = self.halfspace_keys[key_id]

            # Categorical
            t1 = (x_row[I] < U).astype(int)
            t2 = (x_row[I] >= L).astype(int)
            t3 = jnp.prod(jnp.array([t1, t2]), axis=0)
            cat_answers = jnp.prod(t3)

            # Halfspace
            rng_h, rng_b = jax.random.split(halfspace_key, 2)
            hs_mat = (jax.random.normal(rng_h, shape=(numeric_dim,))) / jnp.sqrt(numeric_dim)  # d x h
            b = jax.random.normal(rng_b, shape=(1,))  # 1 x h
            x_row_num_proj = x_row[num_idx]  # n x d
            HS_proj = jnp.dot(x_row_num_proj, hs_mat) - b  # n x h
            above_halfspace = (HS_proj > 0).astype(int)  # n x h

            answers = cat_answers * above_halfspace.squeeze()

            return answers

        if workload_ids is None :
            these_queries = self.queries
        else:
            these_queries = []
            # query_positions = []
            for stat_id in workload_ids:
                a, b = self.workload_positions[stat_id]
                # q_pos = jnp.arange(a, b)
                # query_positions.append(q_pos)
                these_queries.append(self.queries[a:b, :])
            these_queries = jnp.concatenate(these_queries, axis=0)

        temp_stat_fn = jax.vmap(answer_fn, in_axes=(None, 0))

        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, these_queries), None

        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]
        return stat_fn







