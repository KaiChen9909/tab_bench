import chex
import jax.numpy as jnp
import jax
import numpy as np
from typing import Callable
from method.private_gsd.utils import Dataset, Domain, timer
from tqdm import tqdm
from method.private_gsd.stats import AdaptiveStatisticState

cpu = jax.devices("cpu")[0]

class ChainedStatistics:
    all_workloads: list
    selected_workloads: list
    domain: Domain

    modules_workload_fn_jit: list
    modules_all_statistics: list

    def __init__(self, stat_modules: list, rho_allocate = None):
        self.stat_modules = stat_modules # list of Marginal
        self.rho_allocate = rho_allocate

    def fit(self, data: Dataset):
        # X = data.to_numpy()
        self.data = data
        self.N = len(self.data.df)
        self.domain = data.domain
        self.all_workloads = []
        self.modules_workload_fn_jit = []
        self.modules_all_statistics = []
        self.selected_workloads = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod: AdaptiveStatisticState
            stat_mod = self.stat_modules[stat_id]

            data_workload_fn = stat_mod._get_dataset_statistics_fn(jitted=False)

            all_stats = data_workload_fn(data)
            # self.modules_workload_fn_jit.append(jax.jit(stat_mod._get_workload_fn()))
            self.modules_workload_fn_jit.append(stat_mod._get_dataset_statistics_fn(jitted=False))
            self.modules_all_statistics.append(all_stats)
            print(f'\nnumber of queries is {all_stats.shape[0]}\n')

            self.selected_workloads.append([])

        self.all_statistics_fn = self._get_workload_fn()

    def __add_stats(self, stat_id, workload_id, noised_workload_statistics, true_workload_statistics):
        stat_id = int(stat_id)
        workload_id = int(workload_id)
        workload_fn = self.stat_modules[stat_id]._get_workload_fn(workload_ids=[workload_id])
        self.selected_workloads[stat_id].append(
            (workload_id, workload_fn, noised_workload_statistics, true_workload_statistics))

    def __get_selected_workload_ids(self, stat_id: int):
        return jnp.array([tup[0] for tup in self.selected_workloads[stat_id]]).astype(int)
    def get_selected_workload_ids(self, stat_id: int):
        return jnp.array([tup[0] for tup in self.selected_workloads[stat_id]]).astype(int)

    def get_all_true_statistics(self):

        chained_stats = []
        for stat_id in range(len(self.stat_modules)):
            chained_stats.append(self.modules_all_statistics[stat_id])
        return jnp.concatenate(chained_stats)

    def get_all_statistics_fn(self):
        return self._get_workload_fn()

    def get_selected_noised_statistics(self, stat_modules_ids=None):
        if stat_modules_ids is None:
            stat_modules_ids = list(range(len(self.stat_modules)))
        selected_chained_stats = []
        for stat_id in stat_modules_ids:
            temp = [selected[2] for selected in self.selected_workloads[stat_id]]
            if len(temp) > 0:
                temp = jnp.concatenate(temp)
                selected_chained_stats.append(temp)

        return jnp.concatenate(selected_chained_stats)

    def get_selected_statistics_without_noise(self, stat_modules_ids=None):
        if stat_modules_ids is None:
            stat_modules_ids = list(range(len(self.stat_modules)))
        selected_chained_stats = []
        for stat_id in stat_modules_ids:
            temp = [selected[3] for selected in self.selected_workloads[stat_id]]
            if len(temp) > 0:
                temp = jnp.concatenate(temp)
                selected_chained_stats.append(temp)
        return jnp.concatenate(selected_chained_stats)

    def get_selected_statistics_fn(self, stat_modules_ids=None):
        if stat_modules_ids is None:
            stat_modules_ids = list(range(len(self.stat_modules)))
        workload_fn_list = []
        for stat_id in stat_modules_ids:
            stat_mod = self.stat_modules[stat_id]
            workload_ids = self.__get_selected_workload_ids(stat_id)
            if workload_ids.shape[0] > 0:
                workload_fn_list.append(stat_mod._get_workload_fn(workload_ids))

        def chained_workload(X, **kwargs):
            return jnp.concatenate([fn(X, **kwargs) for fn in workload_fn_list], axis=0)

        return chained_workload

    def get_selected_dataset_statistics_fn(self):
        workload_fn_list = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod = self.stat_modules[stat_id]
            workload_ids = self.__get_selected_workload_ids(stat_id)
            if workload_ids.shape[0] > 0:
                workload_fn_list.append(stat_mod._get_dataset_statistics_fn(workload_ids))

        def chained_workload(data: Dataset, **kwargs):
            return jnp.concatenate([fn(data, **kwargs) for fn in workload_fn_list], axis=0)

        return chained_workload

    def get_domain(self):
        return self.domain

    def get_num_workloads(self) -> int:
        s = 0
        for stat_mod in self.stat_modules:
            stat_mod: AdaptiveStatisticState
            s += stat_mod.get_num_workloads()
        return s

    def _get_workload_fn(self) -> Callable:
        workload_fn_list = []
        for stat_mod in self.stat_modules:
            stat_mod: AdaptiveStatisticState
            workload_fn_list.append(stat_mod._get_workload_fn())

        def chained_workload(X):
            return jnp.concatenate([fn(X) for fn in workload_fn_list], axis=0)

        return chained_workload

    def get_dataset_statistics_fn(self, jitted=False):
        workload_fn_list = []
        for stat_mod in self.stat_modules:
            stat_mod: AdaptiveStatisticState
            workload_fn_list.append(stat_mod._get_dataset_statistics_fn(jitted=jitted))

        def chained_workload(data):
            return jnp.concatenate([fn(data) for fn in workload_fn_list], axis=0)

        return chained_workload
    def get_sparse_selected_statistics_fn(self, stat_modules_ids=None, threshold=0):
        if stat_modules_ids is None:
            stat_modules_ids = list(range(len(self.stat_modules)))
        workload_fn_list = []
        selected_chained_stats = []
        for stat_id in stat_modules_ids:
            stat_mod: AdaptiveStatisticState
            stat_mod = self.stat_modules[stat_id]
            temp = [selected[2] for selected in self.selected_workloads[stat_id]]
            workload_ids = self.__get_selected_workload_ids(stat_id)

            query_ids = []
            for workload_id in workload_ids:
                wrk_a, wrk_b = stat_mod._get_workload_positions(workload_id)
                query_ids.append(np.arange(wrk_a, wrk_b))
            query_ids = np.concatenate(query_ids)

            if len(temp) > 0:
                priv_stats = jnp.concatenate(temp)
                sparse_ids = jnp.argwhere(priv_stats > threshold).flatten()
                sparse_stats = priv_stats[sparse_ids]
                selected_chained_stats.append(sparse_stats)
                sparse_query_ids = query_ids[sparse_ids]
                # workload_fn_list.append(stat_mod._get_workload_fn(sparse_workloads_ids))
                workload_fn_list.append(stat_mod._get_stat_fn(sparse_query_ids))


        def chained_workload(X, **kwargs):
            return jnp.concatenate([fn(X, **kwargs) for fn in workload_fn_list], axis=0)

        return jnp.concatenate(selected_chained_stats), chained_workload

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        pass

    def private_measure_all_statistics(self, key: chex.PRNGKey, rho: float, stat_ids: list = None):
        self.selected_workloads = []
        for stat_id in range(len(self.stat_modules)):
            self.selected_workloads.append([])

        # Choose the statistic modules to measure with zCDP
        measure_stats_ids = range(len(self.stat_modules)) if stat_ids is None else stat_ids
        if self.rho_allocate is None:
            if stat_ids is None:
                m = self.get_num_workloads()
            else:
                m = 0
                for stat_id in measure_stats_ids:
                    stat_mod = self.stat_modules[stat_id]
                    m += stat_mod.get_num_workloads()
            rho_per_marginal = [rho / m] * len(self.stat_modules)
        else:
            rho_per_marginal = [self.rho_allocate[stat_id]*rho/self.stat_modules[stat_id].get_num_workloads() for stat_id in measure_stats_ids]

        for stat_id in measure_stats_ids:
            stat_mod = self.stat_modules[stat_id]
            true_stats = self.modules_all_statistics[stat_id]

            for workload_id in range(stat_mod.get_num_workloads()):
                key, key_gaussian = jax.random.split(key, 2)
                wrk_a, wrk_b = stat_mod._get_workload_positions(workload_id)
                stats = true_stats[wrk_a:wrk_b]
                sensitivity = stat_mod._get_workload_sensitivity(workload_id, self.N)
                sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * rho_per_marginal[stat_id])))
                gau_noise = jax.random.normal(key_gaussian, shape=stats.shape) * sigma_gaussian
                selected_noised_stat = jnp.clip(stats + gau_noise, 0, 1)
                # selected_noised_stat = stats + gau_noise
                self.__add_stats(stat_id, workload_id, selected_noised_stat, stats)

    def non_private_measure_all_statistics(self, key: chex.PRNGKey, stat_ids: list = None):
        self.selected_workloads = []
        for stat_id in range(len(self.stat_modules)):
            self.selected_workloads.append([])

        # Choose the statistic modules to measure with zCDP
        measure_stats_ids = range(len(self.stat_modules)) if stat_ids is None else stat_ids

        for stat_id in measure_stats_ids:
            stat_mod = self.stat_modules[stat_id]
            true_stats = self.modules_all_statistics[stat_id]

            for workload_id in range(stat_mod.get_num_workloads()):
                key, key_gaussian = jax.random.split(key, 2)
                wrk_a, wrk_b = stat_mod._get_workload_positions(workload_id)
                stats = true_stats[wrk_a:wrk_b]
                selected_noised_stat = stats
                # selected_noised_stat = stats + gau_noise
                self.__add_stats(stat_id, workload_id, selected_noised_stat, stats)

    def get_sync_data_errors(self, data: Dataset):
        max_errors = []
        for stat_id in range(len(self.stat_modules)):
            stat_mod = self.stat_modules[stat_id]
            module_stat_fn_jit = self.modules_workload_fn_jit[stat_id]

            # Get synthetic data statistics
            module_sync_stats = np.array(module_stat_fn_jit(data))
            # Statistics of original data
            module_true_stats = self.modules_all_statistics[stat_id]

            errors = np.abs(module_true_stats - module_sync_stats)

            stat_max_errors = []
            for workload_id in range(stat_mod.get_num_workloads()):
                # Compute max error for each workload
                wrk_a, wrk_b = stat_mod._get_workload_positions(workload_id)
                max_error = errors[wrk_a:wrk_b].max()
                stat_max_errors.append(max_error)
            max_errors.append(np.array(stat_max_errors))

        # return jnp.array(max_errors)
        return max_errors

    def private_select_measure_statistic(self, key: chex.PRNGKey, rho_per_round: float,
                                         # sync_data_mat: chex.Array,
                                         data: Dataset,
                                         sample_num=1):
        """
        Use this for adaptivity
        :return:
        """
        rho_per_round = rho_per_round / 2

        errors = self.get_sync_data_errors(data)
        # max_sensitivity = max(self.STAT_MODULE.sensitivity)
        key, key_g = jax.random.split(key, 2)
        rs = np.random.RandomState(key_g)
        # Computing Gumbel noise scale based on: https://differentialprivacy.org/one-shot-top-k/

        errors_noise = []
        pos_cnt = 0
        stat_id_pos = []
        workload_id_pos = []
        for stat_id in range(len(self.stat_modules)):
            stat_errors = errors[stat_id]

            noise = rs.gumbel(scale=(np.sqrt(sample_num) / (np.sqrt(2 * rho_per_round) * self.N)),
                              size=stat_errors.shape)
            # noise = jnp.array(noise)
            stat_errors_noise = stat_errors + noise

            w_ids = self.__get_selected_workload_ids(stat_id)
            stat_errors_noise[np.array(w_ids)] = -10000
            # stat_errors_noise = stat_errors_noise.at[w_ids].set(-100000)
            errors_noise.append(stat_errors_noise)

            m = stat_errors_noise.shape[0]
            vec_stat_id = jnp.ones(m) * stat_id
            vec_work_id = jnp.arange(m)
            stat_id_pos.append(vec_stat_id)
            workload_id_pos.append(vec_work_id)

        stat_id_pos = jnp.concatenate(stat_id_pos)
        workload_id_pos = jnp.concatenate(workload_id_pos)
        errors_noise = jnp.concatenate(errors_noise)

        errors_noise_flatten = errors_noise.flatten()
        top_k_indices = (-errors_noise_flatten).argsort()[:sample_num]

        for flatten_id in top_k_indices:
            stat_id = stat_id_pos[flatten_id]
            workload_id = workload_id_pos[flatten_id]

            # stat_id, workload_id = jnp.unravel_index(flatten_ids, errors_noise.shape)
            stat_id = int(stat_id)
            workload_id = int(workload_id)
            gaussian_rho_per_round = rho_per_round / sample_num
            key, key_gaussian = jax.random.split(key, 2)
            stat_mod = self.stat_modules[stat_id]

            # Retrieve workload
            wrk_a, wrk_b = self.stat_modules[stat_id]._get_workload_positions(workload_id)
            stats = self.modules_all_statistics[stat_id][wrk_a:wrk_b]
            sensitivity = stat_mod._get_workload_sensitivity(workload_id, self.N)

            sigma_gaussian = float(np.sqrt(sensitivity ** 2 / (2 * gaussian_rho_per_round)))
            gau_noise = jax.random.normal(key_gaussian, shape=stats.shape) * sigma_gaussian
            selected_noised_stat = jnp.clip(stats + gau_noise, 0, 1)
            self.__add_stats(stat_id, workload_id, selected_noised_stat, stats)

    def get_selected_trimmed_statistics_fn(self, stat_modules_ids=None):
        if stat_modules_ids is None:
            stat_modules_ids = list(range(len(self.stat_modules)))
        workload_fn_list = []
        selected_true_chained_stats = []
        selected_noised_chained_stats = []
        for stat_id in stat_modules_ids:
            stat_mod: AdaptiveStatisticState
            stat_mod = self.stat_modules[stat_id]
            query_ids_list = []
            for workload_id, workload_fn, noised_workload_stats, true_workload_stats in self.selected_workloads[stat_id]:
                S = stat_mod._get_workload_sensitivity(workload_id, 1)**2 / 2
                wrk_a, wrk_b = stat_mod._get_workload_positions(workload_id)
                query_ids = jnp.arange(wrk_a, wrk_b)
                sorted_ids = jnp.argsort(-noised_workload_stats)
                sorted_values = noised_workload_stats[sorted_ids]
                cumsum_sorted_values = jnp.cumsum(sorted_values)
                q_id = jnp.searchsorted(cumsum_sorted_values, S)
                # workload_top_k = min(noised_workload_stats.shape[0], self.max_queries_per_workload)
                topk_ids = sorted_ids[:q_id]
                selected_true_chained_stats.append(true_workload_stats[topk_ids])
                selected_noised_chained_stats.append(noised_workload_stats[topk_ids])
                query_ids_list.append(query_ids[topk_ids])
            query_ids_concat = jnp.concatenate(query_ids_list)
            tmp_fn = stat_mod._get_stat_fn(query_ids_concat)
            workload_fn_list.append(tmp_fn)

        def chained_workload(X, **kwargs):
            return jnp.concatenate([fn(X, **kwargs) for fn in workload_fn_list], axis=0)

        return jnp.concatenate(selected_true_chained_stats), jnp.concatenate(selected_noised_chained_stats), chained_workload

    def reselect_stats(self):
        self.selected_workloads = []
        for stat_id in range(len(self.stat_modules)):
            self.selected_workloads.append([])





def exponential_mechanism(key: jnp.ndarray, scores: jnp.ndarray, eps0: float, sensitivity: float):
    dist = jax.nn.softmax(2 * eps0 * scores / (2 * sensitivity))
    cumulative_dist = jnp.cumsum(dist)
    max_query_idx = jnp.searchsorted(cumulative_dist, jax.random.uniform(key, shape=(1,)))
    return max_query_idx[0]


if __name__ == "__main__":
    from stats import Marginals
    from dev.toy_datasets.classification import get_classification

    data = get_classification()

    # marginal_module1 = Marginals.get_all_kway_combinations(data.domain, k=1, bins=[2, 4])
    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])
    chained_module = ChainedStatistics([marginal_module2])
    chained_module.fit(data)
    chained_module.private_measure_all_statistics(key=jax.random.PRNGKey(0), rho=10)

    chained_module.get_selected_trimmed_statistics_fn()

