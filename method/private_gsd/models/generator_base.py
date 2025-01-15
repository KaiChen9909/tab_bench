import jax
from method.private_gsd.stats import ChainedStatistics
import time
from method.private_gsd.utils import Dataset, timer
from method.private_gsd.utils.cdp2adp import cdp_rho, cdp_eps
import jax.numpy as jnp
from typing import Callable

class Generator:
    data_size: int
    early_stop_elapsed_time = 1
    last_time: float = None
    last_error: float = None
    loss_change_threshold: float = 0.001

    def early_stop_init(self):
        self.last_time: float = time.time()
        self.last_error = 10000000
        self.start_time = time.time()
        self.last_update_iteration = 0

    def early_stop(self, t, error):
        """
        :param t: current iteration
        :param error: the error on this iteration
        :return:
        """
        if self.last_update_iteration == 0:  # If it's the first time don't halt
            self.last_error = error
            self.last_update_iteration = t
            return False

        stop_early = False
        loss_change = (self.last_error - error) / self.last_error
        iterations_increase = (t - self.last_update_iteration) / self.last_update_iteration
        if loss_change > self.loss_change_threshold:
            # If the loss improves same the iteration and the error
            self.last_error = error
            self.last_update_iteration = t

        if iterations_increase > 0.1:
            # If the iterations have increased by more than 10% since the last improvement then halt
            stop_early = True
            self.last_error = error
            self.last_update_iteration = t
        return stop_early

    def fit(self, key: jax.random.PRNGKey, stat: ChainedStatistics, init_data: Dataset = None,
            tolerance: float = 0, adaptive_epoch: int = 1) -> Dataset:
        pass

    def fit_dp(self, key: jax.random.PRNGKey, stat_module: ChainedStatistics, epsilon: float, delta: float,
               init_data: Dataset = None, tolerance: float = 0) -> Dataset:
        rho = cdp_rho(epsilon, delta)
        eps2 = cdp_eps(rho, delta)
        assert eps2 <= epsilon, f'Error: ({eps2})-zCDP -> ({eps2})-DP'
        return self.fit_zcdp(key, stat_module, rho, init_data, tolerance)

    def fit_zcdp(self, key: jax.random.PRNGKey, stat_module: ChainedStatistics, rho: float,
                 init_data: Dataset = None, tolerance: float = 0) -> Dataset:
        key_stats, key_fit = jax.random.split(key)
        stat_module.reselect_stats()
        stat_module.private_measure_all_statistics(key_stats, rho)

        return self.fit(key_fit, stat_module, init_data, tolerance, adaptive_epoch=1)


    def fit_non_private(self, key: jax.random.PRNGKey, stat_module: ChainedStatistics,
                 init_data: Dataset = None, tolerance: float = 0) -> Dataset:
        key_stats, key_fit = jax.random.split(key)
        stat_module.reselect_stats()
        stat_module.non_private_measure_all_statistics(key_stats)

        return self.fit(key_fit, stat_module, init_data, tolerance, adaptive_epoch=1)

    def fit_dp_adaptive(self, key: jax.random.PRNGKey,
                        stat_module: ChainedStatistics, rounds,
                        epsilon: float, delta: float,
                        tolerance: float = 0,
                        start_sync=True,
                        print_progress=True,
                        debug_fn: Callable = None, num_sample=1):
        rho = cdp_rho(epsilon, delta)
        eps2 = cdp_eps(rho, delta)
        assert rho < epsilon, f'Error: ({rho})-zCDP -> ({eps2})-DP'
        return self.fit_zcdp_adaptive(key, stat_module, rounds, rho, tolerance, start_sync, print_progress, debug_fn,
                                      num_sample)

    def fit_zcdp_adaptive(self, key: jax.random.PRNGKey,
                          stat_module: ChainedStatistics,
                          rounds: int,
                          rho: float, tolerance: float = 0,
                          start_sync=False,
                          print_progress=False,
                          debug_fn: Callable = None, num_sample=1):

        # Reset selected statistics
        stat_module.reselect_stats()

        rho_per_round = rho / rounds

        key, key_init = jax.random.split(key, 2)
        init_seed = int(jax.random.randint(key_init, minval=0, maxval=2 ** 20, shape=(1,))[0])
        sync_dataset = Dataset.synthetic(stat_module.get_domain(), N=self.data_size, seed=init_seed)

        for i in range(1, rounds + 1):
            if i < rounds:
                self.loss_change_threshold = 0.01
            else:
                self.loss_change_threshold = 0.001

            # Select a query with max error using the exponential mechanism and evaluate
            select_time = timer()
            # X_sync = sync_dataset.to_numpy()
            key, subkey_select = jax.random.split(key, 2)
            stat_module.private_select_measure_statistic(subkey_select, rho_per_round, sync_dataset, num_sample)
            select_time = timer() - select_time

            fit_time = timer()
            key, key_fit = jax.random.split(key, 2)
            dataset: Dataset
            if start_sync:
                new_sync_dataset = self.fit(key_fit, stat_module, sync_dataset, tolerance=tolerance, adaptive_epoch=i)
            else:
                new_sync_dataset = self.fit(key_fit, stat_module, tolerance=tolerance, adaptive_epoch=i)
            fit_time = timer() - fit_time

            if print_progress:
                # Errors of selected statistics. Debug the success of the project step.
                priv_stats = stat_module.get_selected_noised_statistics()
                selected_true_stats = stat_module.get_selected_statistics_without_noise()
                stat_fn = stat_module.get_selected_dataset_statistics_fn()
                init_round_errors = jnp.abs(selected_true_stats - stat_fn(sync_dataset))
                round_errors = jnp.abs(selected_true_stats - stat_fn(new_sync_dataset))
                gau_error = jnp.abs(selected_true_stats - priv_stats)

                # Get errors for debugging. This is
                all_true_stats = stat_module.get_all_true_statistics()
                all_sync_stat_fn = stat_module.get_dataset_statistics_fn()
                all_sync_stats = all_sync_stat_fn(new_sync_dataset)
                all_errors = jnp.abs(all_true_stats - all_sync_stats)

                print(f'Epoch {i:03}: '
                      f'\tTotal: error(max/avg) is {all_errors.max():.4f}/{all_errors.mean():.7f}.\t ||'
                      f'\tRound init: True error(max/l2) is {init_round_errors.max():.5f}/{init_round_errors.mean():.7f}.'
                      f'\tRound final: True error(max/l2) is {round_errors.max():.5f}/{round_errors.mean():.7f}.'
                      f'\tGaussian error(max/l2) is {gau_error.max():.5f}/{gau_error.mean():.7f}.'
                      f'\tElapsed time(fit/select)={fit_time:>7.3f}/{select_time:.3f}')


            sync_dataset = new_sync_dataset

            if debug_fn is not None:
                debug_fn(i, sync_dataset)

        return sync_dataset

    def fit_dp_hybrid(self, key: jax.random.PRNGKey,
                        stat_module: ChainedStatistics,
                        rounds,
                        epsilon: float, delta: float,
                        tolerance: float = 0,
                        start_sync=True,
                        print_progress=True,
                        debug_fn: Callable = None, num_sample=1,
                      oneshot_share_opt=None):
        rho = cdp_rho(epsilon, delta)
        eps2 = cdp_eps(rho, delta)

        assert rho < epsilon, f'Error: ({rho})-zCDP -> ({eps2})-DP'
        return self.fit_zcdp_hybrid(key, stat_module, rounds, rho, tolerance,
                                    start_sync, print_progress, debug_fn, num_sample, oneshot_share_opt)

    def fit_zcdp_hybrid(self, key: jax.random.PRNGKey,
                            stat_module: ChainedStatistics,
                            rounds: int,
                            rho: float, tolerance: float = 0,
                            start_sync=False,
                            print_progress=False,
                            debug_fn: Callable = None,
                        num_sample=1,
                        oneshot_share_opt=None):
        oneshot_stats_ids = [0]
        num_adaptive_queries = rounds * num_sample
        oneshot_workloads = stat_module.stat_modules[0].get_num_workloads()
        if oneshot_share_opt is not None:
            oneshot_share = oneshot_share_opt
        else:
            oneshot_share = oneshot_workloads / (oneshot_workloads + num_adaptive_queries)
        print(f'oneshot_share={oneshot_share:.4f}')
        # Reset selected statistics
        stat_module.reselect_stats()

        rho_oneshot = oneshot_share * rho
        rho_adaptive = rho - rho_oneshot

        # Oneshot
        key, key_oneshot = jax.random.split(key, 2)
        stat_module.private_measure_all_statistics(key_oneshot, rho_oneshot, stat_ids=oneshot_stats_ids)


        ## Adaptive
        rho_per_round = rho_adaptive / rounds

        key, key_init = jax.random.split(key, 2)
        init_seed = int(jax.random.randint(key_init, minval=0, maxval=2 ** 20, shape=(1,))[0])
        sync_dataset = Dataset.synthetic(stat_module.get_domain(), N=self.data_size, seed=init_seed)

        key, key_oneshot_fit = jax.random.split(key, 2)
        sync_dataset = self.fit(key_oneshot_fit, stat_module, sync_dataset, tolerance=tolerance)

        priv_stats = stat_module.get_selected_noised_statistics()
        selected_true_stats = stat_module.get_selected_statistics_without_noise()
        stat_fn = stat_module.get_selected_dataset_statistics_fn()
        oneshot_error = jnp.abs(selected_true_stats - stat_fn(sync_dataset))
        gau_error = jnp.abs(selected_true_stats - priv_stats)
        print(f'Epoch oneshot: '
              f'\tTrue error(max/l2) is {oneshot_error.max():.5f}/{oneshot_error.mean():.7f}.'
              f'\tGaussian error(max/l2) is {gau_error.max():.5f}/{gau_error.mean():.7f}.')

        for i in range(1, rounds + 1):
            # if i < rounds:
            #     self.loss_change_threshold = 0.01
            # else:
            #     self.loss_change_threshold = 0.001

            # Select a query with max error using the exponential mechanism and evaluate
            select_time = timer()
            # X_sync = sync_dataset.to_numpy()
            key, subkey_select = jax.random.split(key, 2)
            stat_module.private_select_measure_statistic(subkey_select, rho_per_round, sync_dataset, num_sample)
            select_time = timer() - select_time

            fit_time = timer()
            key, key_fit = jax.random.split(key, 2)
            dataset: Dataset
            if start_sync:
                new_sync_dataset = self.fit(key_fit, stat_module, sync_dataset, tolerance=tolerance, adaptive_epoch=i)
            else:
                new_sync_dataset = self.fit(key_fit, stat_module, tolerance=tolerance, adaptive_epoch=i)
            fit_time = timer() - fit_time

            if print_progress:
                # Errors of selected statistics. Debug the success of the project step.
                priv_stats = stat_module.get_selected_noised_statistics()
                selected_true_stats = stat_module.get_selected_statistics_without_noise()
                stat_fn = stat_module.get_selected_dataset_statistics_fn()
                init_round_errors = jnp.abs(selected_true_stats - stat_fn(sync_dataset))
                round_errors = jnp.abs(selected_true_stats - stat_fn(new_sync_dataset))
                gau_error = jnp.abs(selected_true_stats - priv_stats)

                # Get errors for debugging. This is
                all_true_stats = stat_module.get_all_true_statistics()
                all_sync_stat_fn = stat_module.get_dataset_statistics_fn()
                all_sync_stats = all_sync_stat_fn(new_sync_dataset)
                all_errors = jnp.abs(all_true_stats - all_sync_stats)

                print(f'Epoch {i:03}: '
                      f'\tTotal: error(max/avg) is {all_errors.max():.4f}/{all_errors.mean():.7f}.\t ||'
                      f'\tRound init: True error(max/l2) is {init_round_errors.max():.5f}/{init_round_errors.mean():.7f}.'
                      f'\tRound final: True error(max/l2) is {round_errors.max():.5f}/{round_errors.mean():.7f}.'
                      f'\tGaussian error(max/l2) is {gau_error.max():.5f}/{gau_error.mean():.7f}.'
                      f'\tElapsed time(fit/select)={fit_time:>7.3f}/{select_time:.3f}')

            sync_dataset = new_sync_dataset

            if debug_fn is not None:
                debug_fn(i, sync_dataset)

        return sync_dataset