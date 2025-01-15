import jax.numpy as jnp
import pandas as pd
from method.private_gsd.models import Generator
import time
from method.private_gsd.stats import ChainedStatistics
import jax
import chex
from flax import struct
from method.private_gsd.utils import Dataset, Domain, timer
from functools import partial
from typing import Tuple


@struct.dataclass
class EvoState:
    archive: chex.Array
    fitness: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max


@struct.dataclass
class PopulationState:
    X: chex.Array
    remove_row: chex.Array
    add_row: chex.Array


"""
Implement crossover that is specific to synthetic data
"""


def get_best_fitness_member(
    x: chex.Array, fitness: chex.Array, state
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    best_in_gen = jnp.argmin(fitness)
    best_in_gen_fitness, best_in_gen_member = (
        fitness[best_in_gen],
        x[best_in_gen],
    )
    replace_best = best_in_gen_fitness < state.best_fitness
    best_fitness = jax.lax.select(
        replace_best, best_in_gen_fitness, state.best_fitness
    )
    best_member = jax.lax.select(
        replace_best, best_in_gen_member, state.best_member
    )
    return best_member, best_fitness, replace_best, best_in_gen

class SimpleGAforSyncData:
    def __init__(self, domain: Domain,
                 data_size: int,
                 population_size_muta: int = 50,
                 population_size_cross: int = 50,
                 population_size: int = None,
                 elite_size: int = 5,
                 muta_rate: int = 1,
                 mate_rate: int = 1,
                 debugging=False):
        """Simple Genetic Algorithm For Synthetic Data Search Adapted from (Such et al., 2017)
        Reference: https://arxiv.org/abs/1712.06567
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""

        if population_size is not None:
            self.population_size_muta = population_size // 2
            self.population_size_cross = population_size // 2
        else:
            self.population_size_muta = population_size_muta
            self.population_size_cross = population_size_cross
        self.population_size = self.population_size_muta + self.population_size_cross
        self.elite_size = elite_size
        self.data_size = data_size

        self.domain = domain
        self.strategy_name = "SimpleGA"
        self.num_devices = jax.device_count()
        self.domain = domain

        # It's recommended to always set  muta_rate=muta_rate=1
        assert muta_rate > 0, "Mutation rate must be greater than zero."
        assert mate_rate > 0, "Mate rate must be greater than zero."
        assert muta_rate == mate_rate, "Mutations and crossover must be the same."
        self.muta_rate = muta_rate
        self.mate_rate = mate_rate
        self.debugging = debugging

    def initialize(
            self, rng: chex.PRNGKey
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        init_x = self.initialize_elite_population(rng)
        state = EvoState(
            archive=init_x.astype(jnp.float32),
            fitness=jnp.zeros(self.elite_size) + jnp.finfo(jnp.float32).max,
            best_member=init_x[0].astype(jnp.float32),
            best_fitness=jnp.finfo(jnp.float32).max
        )

        rng1, rng2 = jax.random.split(rng, 2)
        random_numbers = jax.random.permutation(rng1, self.data_size, independent=True)
        muta_fn = get_mutate_fn(muta_rate=self.muta_rate, random_numbers=random_numbers)
        mate_fn = get_mating_fn(self.domain, mate_rate=self.mate_rate, random_numbers=random_numbers)

        self.muta_vmap = jax.jit(jax.vmap(muta_fn, in_axes=(None, 0, 0)))
        self.mate_vmap = jax.jit(jax.vmap(mate_fn, in_axes=(None, 0, 0)))

        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_elite_population(self, rng: chex.PRNGKey):
        d = len(self.domain.attrs)
        pop = Dataset.synthetic_jax_rng(self.domain, self.elite_size * self.data_size, rng)
        initialization = pop.reshape((self.elite_size, self.data_size, d))
        return initialization

    @partial(jax.jit, static_argnums=(0,))
    def initialize_random_population(self, rng: chex.PRNGKey):
        pop = Dataset.synthetic_jax_rng(self.domain, self.population_size, rng)
        return pop

    def ask(self, rng: chex.PRNGKey, state: EvoState):
        rng_pop, rng_ask = jax.random.split(rng, 2)
        random_data = self.initialize_random_population(rng_pop)
        return self.ask_strategy(rng_ask, random_data, state)

    @partial(jax.jit, static_argnums=(0,))
    def ask_strategy(self, rng: chex.PRNGKey, random_data, state: EvoState):
        rng, rng_i, rng_j, rng_k, rng_muta, rng_mate, rng_mutate = jax.random.split(rng, 7)

        random_data = random_data[:self.population_size_muta, :]
        rng_muta_split = jax.random.split(rng_muta, self.population_size_muta)
        pop_muta = self.muta_vmap(state.best_member, rng_muta_split, random_data)

        rng_mate_split = jax.random.split(rng_mate, self.population_size_cross)
        j = jax.random.randint(rng_j, minval=0, maxval=self.elite_size, shape=(self.population_size_cross,))
        x_j = state.archive[j]
        pop_mate = self.mate_vmap(state.best_member, rng_mate_split, x_j)

        remove_row = jnp.concatenate((pop_muta.remove_row, pop_mate.remove_row), axis=0)
        add_row = jnp.concatenate((pop_muta.add_row, pop_mate.add_row), axis=0)
        population = PopulationState(
            X=jnp.concatenate((pop_muta.X, pop_mate.X)),
            remove_row=remove_row,
            add_row=add_row)
        return population

    @partial(jax.jit, static_argnums=(0,))
    def tell(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
    ) -> Tuple[EvoState, chex.Array, chex.Array]:
        """`tell` performance data for strategy state update."""
        state = self.tell_strategy(x, fitness, state)
        best_member, best_fitness, replace_best, best_in_gen = get_best_fitness_member(x, fitness, state)
        return state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
        ), replace_best, best_in_gen

    def tell_strategy(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
    ) -> EvoState:
        fitness_concat = jnp.concatenate([state.fitness, fitness])
        solution_concat = jnp.concatenate([state.archive, x])
        idx = jnp.argsort(fitness_concat)[0: self.elite_size]

        new_fitness = fitness_concat[idx]
        new_archive = solution_concat[idx]

        new_state = state.replace(
            fitness=new_fitness, archive=new_archive,
        )

        return new_state


def get_mutate_fn(muta_rate: int, random_numbers):
    def muta(
            X0,
            rng: chex.PRNGKey,  initialization
    ) -> PopulationState:
        X0 = X0.astype(jnp.float32)

        n, d = X0.shape
        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng_temp, rng_normal = jax.random.split(rng, 9)

        temp = jax.random.randint(rng1, minval=0, maxval=random_numbers.shape[0] - muta_rate, shape=(1,))
        temp_id = jnp.arange(muta_rate) + temp
        temp_id = jax.random.permutation(rng2, random_numbers[temp_id], independent=True)
        mut_rows = temp_id

        ###################
        ## Mutate
        mut_col = jax.random.randint(rng5, minval=0, maxval=d, shape=(muta_rate,))
        values = initialization[mut_col]

        removed_rows_muta = X0[mut_rows, :].reshape((muta_rate, d))
        X_mut = X0.at[mut_rows, mut_col].set(values)
        added_rows_muta = X_mut[mut_rows, :].reshape((muta_rate, d))

        pop_state = PopulationState(X=X_mut, remove_row=removed_rows_muta, add_row=added_rows_muta)
        return pop_state

    return muta

def get_mating_fn(domain: Domain, mate_rate: int, random_numbers):
    d = len(domain.attrs)
    numeric_idx = domain.get_attribute_indices(domain.get_numeric_cols()).astype(int)
    mask = jnp.zeros(d)
    mask = mask.at[numeric_idx].set(1)
    mask = mask.reshape((1, d))

    def mate(
            X0, rng: chex.PRNGKey, elite_rows: chex.Array
    ) -> PopulationState:
        X0 = X0.astype(jnp.float32)
        elite_rows = elite_rows.astype(jnp.float32)

        n, d = X0.shape
        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng_temp, rng_normal = jax.random.split(rng, 9)

        # Choose row indices to cross
        temp = jax.random.randint(rng1, minval=0, maxval=random_numbers.shape[0] - mate_rate, shape=(1,))
        temp_id = jnp.arange(mate_rate) + temp
        temp_id = jax.random.permutation(rng2, random_numbers[temp_id], independent=True)
        remove_rows_idx = temp_id[:mate_rate]

        removed_rows_mate = X0[remove_rows_idx, :].reshape((mate_rate, d))
        add_rows_idx = jax.random.randint(rng3, minval=0, maxval=elite_rows.shape[0], shape=(mate_rate,))

        # Copy this row onto the dataset
        new_rows = elite_rows[add_rows_idx]
        noise = mask * jax.random.normal(rng_normal, shape=(new_rows.shape[0], d))
        new_rows = new_rows + noise
        new_rows = new_rows.at[:, numeric_idx].set(jnp.clip(new_rows[:, numeric_idx], 0, 1))

        # Only crossover a subset of the values in the rows
        rng_mate1, rng_mate2 = jax.random.split(rng_temp, 2)
        temp = jnp.repeat(jnp.array([1, 0]), jnp.array([1, d - 1]), total_repeat_length=d)
        temp = jnp.repeat(temp.reshape(1, -1), new_rows.shape[0])
        temp = jax.random.permutation(rng_mate2, temp).reshape((mate_rate, -1))

        added_rows_mate = temp * new_rows + (1 - temp) * removed_rows_mate

        # Copy new rows
        X = X0.at[remove_rows_idx].set(added_rows_mate)

        pop_state = PopulationState(X=X, remove_row=removed_rows_mate, add_row=added_rows_mate)
        return pop_state

    return mate

######################################################################
######################################################################
######################################################################
######################################################################


# @dataclass
class GSD(Generator):

    def __init__(self,
                 domain,
                 population_size_muta=50,
                 population_size_cross=50,
                 population_size=None,
                 muta_rate=1,
                 mate_rate=1,
                 print_progress=False,
                 stop_early=True,
                 stop_early_gen=None,
                 stop_eary_threshold=0,
                 sparse_statistics=False, 
                 num_encoder = None,
                 num_idx = None,
                 args = None
                 ):
        self.domain = domain
        self.print_progress = print_progress
        self.stop_early = stop_early
        self.stop_early_gen = stop_early_gen
        self.stop_eary_threshold = stop_eary_threshold
        self.sparse_statistics = sparse_statistics
        self.population_size_muta = population_size_muta
        self.population_size_cross = population_size_cross
        self.population_size = population_size
        self.muta_rate = muta_rate
        self.mate_rate = mate_rate
        self.stop_generation = None
        self.num_encoder = num_encoder
        self.num_idx = num_idx
        self.args = args

    def __str__(self):
        return f'GSD'
    
    def zcdp_syn_init(self, key: jax.random.PRNGKey, stat_module: ChainedStatistics, rho: float,
                 init_data: Dataset = None, tolerance: float = 0):
        self.key = key
        self.stat_module = stat_module 
        self.rho = rho 
        self.tolerance = tolerance
        self.init_data = init_data 
    

    def syn(self, n_sample, preprocesser, parent_dir=None):
        oversample = 1 if n_sample < 10000 else n_sample/10000
        n_sample = n_sample if n_sample < 10000 else 10000

        self.num_generations = n_sample*10
        self.data_size = n_sample
        self.stop_early_min_generation = self.stop_early_gen if self.stop_early_gen is not None else n_sample
        self.strategy = SimpleGAforSyncData(self.domain, n_sample,
                                            population_size_muta=self.population_size_muta,
                                            population_size_cross=self.population_size_cross,
                                            population_size=self.population_size,
                                            muta_rate=self.muta_rate, mate_rate=self.mate_rate)


        key_stats, key_fit = jax.random.split(self.key)
        self.stat_module.reselect_stats()
        self.stat_module.private_measure_all_statistics(key_stats, self.rho)

        synth = self.fit(key_fit, self.stat_module, self.init_data, self.tolerance, adaptive_epoch=1)
        
        if self.num_encoder is not None:
            synth.df.iloc[:, self.num_idx] = self.num_encoder.inverse_transform(synth.df.iloc[:, self.num_idx])

        if oversample > 1:
            synthesized_df = self.oversample(synth.df, oversample)
        else:
            synthesized_df = synth.df

        if parent_dir is not None:
            preprocesser.reverse_data(synthesized_df, parent_dir) 

        return synthesized_df


    def oversample(self, df, oversample=5):
        size = int(df.shape[0] * oversample) 
        new_df = df.sample(n=size, replace=True)
        return new_df


    def fit(self, key, adaptive_statistic: ChainedStatistics,
            sync_dataset: Dataset = None, tolerance: float = 0.0, adaptive_epoch=1):
        """
        Minimize error between real_stats and sync_stats
        """

        self.stop_generation = None
        init_time = timer()

        if self.sparse_statistics:
            selected_statistics, selected_noised_statistics, statistics_fn = adaptive_statistic.get_selected_trimmed_statistics_fn()
            if self.print_progress:
                print(f'Number of sparse statistics is {selected_statistics.shape[0]}. Time = {timer() - init_time:.2f}')
        else:
            selected_noised_statistics = adaptive_statistic.get_selected_noised_statistics()
            selected_statistics = adaptive_statistic.get_selected_statistics_without_noise()
            statistics_fn = adaptive_statistic.get_selected_statistics_fn()

        # For debugging
        @jax.jit
        def true_loss(X_arg):
            error = jnp.abs(selected_statistics - statistics_fn(X_arg))
            return jnp.abs(error).max(), jnp.abs(error).mean(), jnp.linalg.norm(error, ord=2)

        @jax.jit
        def private_loss(X_arg):
            error = jnp.abs(selected_noised_statistics - statistics_fn(X_arg))
            return jnp.abs(error).max(), jnp.abs(error).mean(), jnp.linalg.norm(error, ord=2)

        def fitness_fn(stats: chex.Array, pop_state: PopulationState):
            # Process one member of the population
            # 1) Update the statistics of this synthetic dataset
            rem_row = pop_state.remove_row
            add_row = pop_state.add_row
            num_rows = rem_row.shape[0]
            add_stats = (num_rows * statistics_fn(add_row))
            rem_stats = (num_rows * statistics_fn(rem_row))
            upt_sync_stat = stats.reshape(-1) + add_stats - rem_stats
            # 2) Compute its fitness based on the statistics
            fitness = jnp.linalg.norm(selected_noised_statistics - upt_sync_stat / self.data_size, ord=2) ** 2
            return fitness

        fitness_fn_vmap = jax.vmap(fitness_fn, in_axes=(None, 0))
        fitness_fn_jit = jax.jit(fitness_fn_vmap)


        # INITIALIZE STATE
        key, subkey = jax.random.split(key, 2)

        state = self.strategy.initialize(subkey)

        if sync_dataset is not None:
            init_sync = sync_dataset.to_numpy()
            temp = init_sync.reshape((1, init_sync.shape[0], init_sync.shape[1]))
            new_archive = jnp.concatenate([temp, state.archive[1:, :, :]])
            state = state.replace(archive=new_archive)

        elite_population_fn = jax.vmap(statistics_fn, in_axes=(0,))
        elite_fitness = jnp.linalg.norm(selected_noised_statistics - elite_population_fn(state.archive), axis=1,
                                        ord=2) ** 2
        best_member_id = elite_fitness.argmin()
        state = state.replace(
            fitness=elite_fitness,
            best_member=state.archive[best_member_id],
            best_fitness=elite_fitness[best_member_id]
        )

        self.early_stop_init()  # Initiate time-based early stop system

        best_fitness_total = 100000
        ask_time = 0
        elite_stat_time = 0
        fit_time = 0
        tell_time = 0
        last_fitness = None
        self.fitness_record = []

        if self.print_progress:
            timer(init_time, '\tSetup time = ')

        elite_stat = self.data_size * statistics_fn(state.best_member)  # Statistics of best SD

        def update_elite_stat(elite_stat_arg,
                              population_state: PopulationState,
                              replace_best,
                              best_id_arg
                              ):
            num_rows = population_state.remove_row[0].shape[0]

            new_elite_stat = jax.lax.select(
                    replace_best,
                    elite_stat_arg
                        - (num_rows * statistics_fn(population_state.remove_row[best_id_arg]))
                        + (num_rows * statistics_fn(population_state.add_row[best_id_arg])),
                    elite_stat_arg
                )
            return new_elite_stat

        update_elite_stat_jit = jax.jit(update_elite_stat)
        LAST_LAG_FITNESS = state.best_fitness
        true_results = []
        for t in range(self.num_generations):
            self.stop_generation = t  # Update the stop generation
            # ASK
            t0 = timer()
            key, ask_subkey = jax.random.split(key, 2)
            population_state = self.strategy.ask(ask_subkey, state)
            population_state.remove_row.block_until_ready()
            ask_time += timer() - t0

            # FIT
            t0 = timer()
            fitness = fitness_fn_jit(elite_stat, population_state).block_until_ready()
            fit_time += timer() - t0

            # TELL
            t0 = timer()
            state, rep_best, best_id = self.strategy.tell(population_state.X, fitness, state)
            state.archive.block_until_ready()
            best_fitness = state.best_fitness

            self.fitness_record.append([t, best_fitness, timer() - init_time])

            tell_time += timer() - t0
            # UPDATE elite_states
            elite_stat = update_elite_stat_jit(elite_stat, population_state, rep_best, best_id).block_until_ready()
            elite_stat_time += timer() - t0

            if best_fitness < self.stop_eary_threshold: break
            if (t % self.stop_early_min_generation) == 0 and t > self.stop_early_min_generation and self.stop_early:
                loss_change = jnp.abs(LAST_LAG_FITNESS - state.best_fitness) / LAST_LAG_FITNESS

                if loss_change < 0.0001:
                    if self.print_progress: print(f'\t\t ### Stop early at {t} ###')
                    break
                LAST_LAG_FITNESS = state.best_fitness

            if (t % 50 == 0) and self.print_progress:
                # DEBUG
                best_fitness_total = min(best_fitness_total, best_fitness)
                if last_fitness is None or best_fitness_total < last_fitness * 0.99 or t > self.num_generations - 2:
                    elapsed_time = timer() - init_time
                    X_sync = state.best_member
                    print(f'\tGen {t:05}, fit={best_fitness_total:.6f}, ', end=' ')
                    t_inf, t_avg, p_l2 = true_loss(X_sync)
                    true_results.append([t, float(t_inf), float(t_avg), float(p_l2)])
                    print(f'\ttrue error(max/avg/l2)=({t_inf:.5f}/{t_avg:.7f}/{p_l2:.3f})', end='')
                    print(f'\t|time={elapsed_time:.4f}(s):', end='')
                    print(f'\task={ask_time:<3.3f}(s), fit={fit_time:<3.3f}(s), tell={tell_time:<3.3f}, ', end='')
                    print(f'elite_stat={elite_stat_time:<3.3f}(s)\t', end='')
                    print()
                    last_fitness = best_fitness_total

        # Save progress for debugging.
        self.true_results_df = pd.DataFrame(true_results, columns=['G', 'Max', 'Avg', 'L2'])
        X_sync = state.best_member
        sync_dataset = Dataset.from_numpy_to_dataset(self.domain, X_sync)
        return sync_dataset

