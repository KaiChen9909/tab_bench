import itertools
import jax
import jax.numpy as jnp
import chex
import matplotlib.pyplot as plt
import pandas as pd

from method.private_gsd.utils import Dataset, Domain
from method.private_gsd.stats import AdaptiveStatisticState
from tqdm import tqdm
import numpy as np


class HalfspacesBT(AdaptiveStatisticState):

    def __init__(self, domain, key: chex.PRNGKey, random_proj: int, bins=(32,)):
        self.domain = domain
        self.key = key
        self.random_proj = random_proj
        self.bins = list(bins)
        self.workload_positions = []
        self.workload_sensitivity = []

        self.cat_pos = domain.get_attribute_indices(domain.get_categorical_cols())
        self.cat_sz = [domain.size(cat) for cat in domain.get_categorical_cols()]
        self.num_pos = domain.get_attribute_indices(domain.get_numeric_cols())

        self.set_up_stats()

    def __str__(self):
        return f'Halfspaces-BT'

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
        key = self.key

        queries = []
        self.workload_positions = []
        self.proj_keys = jax.random.split(key)
        for proj_id in tqdm(range(self.random_proj), desc='Setting up Halfspaces-BT.'):
            # key, key_sub = jax.random.split(key)

            # indices = self.domain.get_attribute_indices(marginal)
            # bins = self.bins if self.is_workload_numeric(marginal) else [-1]
            start_pos = len(queries)
            for bin in self.bins:
                intervals = []
                upper = np.linspace(-1, 1, num=bin+1)[1:]
                lower = np.linspace(-1, 1, num=bin+1)[:-1]
                lower[0] = -100.01
                upper[-1] = 100.01
                # upper = upper.at[-1].set(1.01)
                interval = list(np.vstack((upper, lower)).T)
                intervals.append(interval)
                for v in itertools.product(*intervals):
                    v_arr = np.array(v)
                    up = v_arr.flatten()[::2]
                    lo = v_arr.flatten()[1::2]
                    q = np.concatenate((jnp.array([proj_id]), up, lo))
                    queries.append(q)  # (key), ((a1, a2), (b1, b2))
            end_pos = len(queries)
            self.workload_positions.append((start_pos, end_pos))
            self.workload_sensitivity.append(jnp.sqrt(2 * len(self.bins)))

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

    def random_linear_proj(self, key: chex.PRNGKey, x: chex.Array):
        sum = 0
        norm = jnp.sqrt(x.shape[0])
        for pos, sz in zip(self.cat_pos, self.cat_sz):
            key, key_sub = jax.random.split(key)
            h = jax.random.normal(key_sub, shape=(sz, )) / norm

            x_val = x[pos].astype(int)
            sum += h[x_val]

        key, key_sub = jax.random.split(key)
        h = jax.random.normal(key_sub, shape=(self.num_pos.shape[0], )) / norm
        num_proj = x[self.num_pos]
        sum += jnp.dot(h, num_proj)
        return sum


    def answer_fn(self, x_row: chex.Array, query_single: chex.Array):
        key_id = query_single[0].astype(jnp.uint32)
        key = self.proj_keys[key_id]
        x_proj = self.random_linear_proj(key, x_row)
        hi = query_single[1]
        lo = query_single[2]
        answers1 = lo < x_proj
        answers2 = x_proj < hi
        return (answers1 * answers2).astype(float)

    def _get_stat_fn(self, query_ids):

        these_queries = self.queries[query_ids]
        temp_stat_fn = jax.vmap(self.answer_fn, in_axes=(None, 0))

        def scan_fun(carry, x):
            return carry + temp_stat_fn(x, these_queries), None
        def stat_fn(X):
            out = jax.eval_shape(temp_stat_fn, X[0], these_queries)
            stats = jax.lax.scan(scan_fun, jnp.zeros(out.shape, out.dtype), X)[0]
            return stats / X.shape[0]
        return stat_fn

    # @staticmethod
    # def get_kway_categorical(domain: Domain, k):
    #     kway_combinations = [list(idx) for idx in itertools.combinations(domain.get_categorical_cols(), k)]
    #     return Marginals(domain, kway_combinations, k, bins=[2])
    #
    # @staticmethod
    # def get_all_kway_combinations(domain, k, bins=(32,)):
    #     kway_combinations = [list(idx) for idx in itertools.combinations(domain.attrs, k)]
    #     return Marginals(domain, kway_combinations, k, bins=bins)
    #
    # @staticmethod
    # def get_all_kway_mixed_combinations_v1(domain, k, bins=(32,)):
    #     num_numeric_feats = len(domain.get_numeric_cols())
    #     k_real = num_numeric_feats
    #     kway_combinations = []
    #     for cols in itertools.combinations(domain.attrs, k):
    #         count_disc = 0
    #         count_real = 0
    #         for c in list(cols):
    #             if c in domain.get_numeric_cols():
    #                 count_real += 1
    #             else:
    #                 count_disc += 1
    #         if count_disc > 0 and count_real > 0:
    #             kway_combinations.append(list(cols))
    #
    #     return Marginals(domain, kway_combinations, k, bins=bins)


def get_linear_proj(domain: Domain):
    cat_pos = domain.get_attribute_indices(domain.get_categorical_cols())
    cat_sz = [domain.size(cat) for cat in domain.get_categorical_cols()]
    num_pos = domain.get_attribute_indices(domain.get_numeric_cols())

    def random_linear_proj(key: chex.PRNGKey, x: chex.Array):
        sum = 0
        norm = jnp.sqrt(x.shape[0])
        for pos, sz in zip(cat_pos, cat_sz):
            key, key_sub = jax.random.split(key)
            # h = jax.random.normal(key_sub, shape=(sz, )) / norm
            h = jax.random.randint(key_sub, minval=-1, maxval=2, shape=(sz, )) / norm
            x_val = x[pos].astype(int)
            sum += h[x_val]

        key, key_sub = jax.random.split(key)
        # h = jax.random.normal(key_sub, shape=(num_pos.shape[0], )) / norm
        h = jax.random.randint(key_sub, minval=-1, maxval=2, shape=(num_pos.shape[0], )) / norm
        num_proj = x[num_pos]
        sum += jnp.dot(h, num_proj)
        return sum
    return random_linear_proj



def get_proj(domain: Domain):
    linear_proj = get_linear_proj(domain)
    linear_proj_vmap = jax.vmap(linear_proj, in_axes=(0, None))

    def proj_fn(key, x):
        m = 10
        key, key_lp = jax.random.split(key)
        keys = jax.random.split(key_lp, m)
        proj = linear_proj_vmap(keys, x)

        proj_relu = jax.nn.relu(proj)

        key, key_h = jax.random.split(key)
        h = jax.random.normal(key_h, shape=(m, )) / jnp.sqrt(m)

        return jnp.dot(h, proj_relu)

    return proj_fn


from dp_data import load_domain_config, load_df, ml_eval
import seaborn as sns
def test_proj():
    dataset_name = 'folktables_2018_coverage_CA'
    root_path = '../dp-data-dev/datasets/preprocessed/folktables/1-Year/'
    config = load_domain_config(dataset_name, root_path=root_path)
    df_train = load_df(dataset_name, root_path=root_path, idxs_path='seed0/train')
    df_test = load_df(dataset_name, root_path=root_path, idxs_path='seed0/test')



    print(f'train size: {df_train.shape}')
    print(f'test size:  {df_test.shape}')
    domain = Domain.fromdict(config)
    data = Dataset(df_train.sample(2000), domain)

    X = data.to_numpy()

    sync_cov = pd.read_csv('/Users/vietr002/Code/evolutionary_private_synthetic_data/dev/ICML/sync_data/folktables_2018_coverage_CA/GSD/Ranges/50/10/1.00/sync_data_0.csv')
    sync_data = Dataset(sync_cov, domain)
    # sync_data = Dataset.synthetic(domain, N=2000, seed=0)
    X_sync = sync_data.to_numpy()

    proj_fn_0 = get_proj(domain)
    proj_fn = jax.jit(jax.vmap(proj_fn_0, in_axes=(None, 0)))

    key = jax.random.PRNGKey(2)


    df_list = []
    for i in range(10):
        print(f'proj {i}')
        key, key0 = jax.random.split(key)
        key1, key2 = jax.random.split(key0)
        proj_data_1 = proj_fn(key1, X)
        proj_data_2 = proj_fn(key2, X)

        df_real = pd.DataFrame(np.column_stack((proj_data_1, proj_data_2)), columns=['A', 'B'])
        df_real['Type'] = 'Real'
        df_real['Proj'] = i

        proj_data_sync_1 = proj_fn(key1, X_sync)
        proj_data_sync_2 = proj_fn(key2, X_sync)


        df_sync = pd.DataFrame(np.column_stack((proj_data_sync_1, proj_data_sync_2)), columns=['A', 'B'])
        df_sync['Type'] = 'Sync'
        df_sync['Proj'] = i


        df_list.append(df_real)
        df_list.append(df_sync)

    df = pd.concat(df_list)

    sns.relplot(data=df, x='A', y='B', row='Proj', col='Type', alpha=0.1)
    plt.show()


if __name__ == "__main__":
    test_proj()

