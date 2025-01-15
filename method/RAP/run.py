import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import numpy as jnp
from sklearn.linear_model import LogisticRegression
from method.RAP.dataloading.dataset import Dataset
from method.RAP.dataloading.domain import Domain
from method.RAP.dataloading.dataloading_util import get_upsampled_dataset_from_relaxed
# from benchmark.benchmark_multi_ML import evaluate_machine_learning_task
from method.RAP.mechanisms.mechanism_base import BaseMechanism

RELAXED_SYNC_DATA_PATH = "results/sync_data"


def save_dprime_relaxed_helper(D_prime_relaxed: jnp.array, path_name):
    """
    Save the synthetic data and return path
    """
    D_prime_relaxed = jnp.array(D_prime_relaxed)[:, :]
    jnp.save(path_name, D_prime_relaxed)


def save_synthetic_dataset(
    algorithm_name,
    dataset_name,
    eps,
    params,
    algo_seed,
    D_prime_relaxed,
    synthetic_df: pd.DataFrame,
    runtime: float,
):
    """Save D'"""
    sync_data_path = RELAXED_SYNC_DATA_PATH
    os.makedirs(sync_data_path, exist_ok=True)
    sync_data_path_1 = os.path.join(sync_data_path, algorithm_name)
    os.makedirs(sync_data_path_1, exist_ok=True)
    sync_data_path_2 = os.path.join(sync_data_path_1, dataset_name)
    os.makedirs(sync_data_path_2, exist_ok=True)
    sync_data_path_3 = os.path.join(sync_data_path_2, f"{eps:.2f}")
    os.makedirs(sync_data_path_3, exist_ok=True)
    sync_data_path_4 = os.path.join(sync_data_path_3, f"{params}")
    os.makedirs(sync_data_path_4, exist_ok=True)
    sync_data_path_5 = os.path.join(sync_data_path_4, f"{algo_seed}")
    os.makedirs(sync_data_path_5, exist_ok=True)

    relaxed_file_name = f"relaxed.npy"
    post_file_name = f"synthetic.csv"

    if D_prime_relaxed is not None:
        jnp.save(
            os.path.join(sync_data_path_5, relaxed_file_name),
            jnp.array(D_prime_relaxed)[:, :],
        )
    print(f"Saving synthetic data at {os.path.join(sync_data_path_5, post_file_name)}")
    synthetic_df.to_csv(os.path.join(sync_data_path_5, post_file_name), index=False)

    runtime_file = os.path.join(sync_data_path_5, "runtime.txt")
    print(f"Saving at {runtime_file}")
    f = open(runtime_file, "w")
    f.write(f"{runtime}")
    f.close()


def get_syn_df(
        D_relaxed, domain, oversample_rate=1, seed=0
    ) -> Dataset:
        """
        This function takes as input a relaxed dataset matrix. Then creates a DataFrame in the original
            datas format.
        """
        D_prime_post_dataset = get_upsampled_dataset_from_relaxed(
            D_relaxed, domain, oversample=oversample_rate, seed=seed
        )
        return D_prime_post_dataset.df


class rap_generator():
    def __init__(self, rho, seed, domain, mechanism, args, num_encoder, num_idx):
        self.rho = rho 
        self.seed = seed 
        self.domain = domain 
        self.mechanism = mechanism
        self.args = args
        self.num_encoder = num_encoder 
        self.num_idx = num_idx
    
    def syn(self, n_sample, preprocesser=None, parent_dir=None): 
        if self.args.dataset == 'loan':
            self.oversamples = 40 
        elif self.args.dataset == 'higgs-small':
            self.oversamples = 20
        else:
            self.oversamples = 10
        # self.oversamples = int(n_sample/20000) + 1
        self.mechanism.train(rho=self.rho, seed=self.seed, num_generated_points=n_sample//self.oversamples)
        # self.mechanism.train(rho=self.rho, seed=self.seed, num_generated_points=20000)
        D_prime_relaxed = self.mechanism.get_dprime()[:, :]
        D_prime_original_format_df = (
            get_syn_df(
                D_prime_relaxed, seed=self.seed, oversample_rate=self.oversamples, domain=self.domain
            )
        )

        D_prime_original_format_df.iloc[:, self.num_idx] = self.num_encoder.inverse_transform(D_prime_original_format_df.iloc[:, self.num_idx])

        if parent_dir is not None:
            preprocesser.reverse_data(D_prime_original_format_df, parent_dir)
        return D_prime_original_format_df



def run_experiment(
    mechanism: BaseMechanism,
    df,
    domain,
    rho: float,
    algorithm_seed: int,
    # params: str,
    args=None,
    num_encoder=None,
    num_idx=None,
    oversamples=10
    # ,save_sycn_data=True
):
    """Runs RAP++ and saves the relaxed synthetic data as .npy."""

    algo_seed = algorithm_seed

    mechanism_name = str(mechanism)

    # for algo_seed in algorithm_seed:
        # dataset_container = dataset_fn(algo_seed)
        # true_dataset_post_df = dataset_container.from_dataset_to_df_fn(
        #     dataset_container.train
        # )
        # true_test_dataset_post_df = dataset_container.from_dataset_to_df_fn(
        #     dataset_container.test
        # )
        # cat_cols = dataset_container.cat_columns
        # num_cols = dataset_container.num_columns
        # labels = dataset_container.label_column
        # feature_columns = list(set(cat_cols + num_cols) - set(labels))

        # """ ML Test """
        # # if run_ml_eval:
        # #     for label in dataset_container.label_column:
        # #         print(f'Original results for {label}:')
        # #         evaluate_machine_learning_task(true_dataset_post_df,
        # #                          true_test_dataset_post_df,
        # #                         feature_columns=feature_columns,
        # #                          label_column=label,
        # #                          cat_columns=dataset_container.cat_columns,
        # #                          endmodel=LogisticRegression(penalty='l1', solver="liblinear")
        # #                          )

        # debug_fn = get_debug_fn(dataset_container) if get_debug_fn is not None else None

    domain = Domain.fromdict(domain, targets = ['y_attr'])
    dataset = Dataset(df, domain)


    mechanism.initialize(
        dataset, algo_seed
    )  # Pass the dataset to algorithm so that it can create the statistics.

        # print(
        #     f"\n\n\nTraining {str(mechanism)} with rho={rho}, algo_seed={algo_seed}"
        # )

    generator = rap_generator(
        rho = rho,
        seed = algo_seed,
        domain = domain, 
        mechanism = mechanism,
        args = args,
        num_encoder=num_encoder,
        num_idx=num_idx
    )
    return generator

        # """ Train RAP """
        # stime = time.time()
        # mechanism.train(rho=rho, seed=algo_seed)
        # runtime_seconds = time.time() - stime
        # minutes, seconds = divmod(time.time() - stime, 60)
        # print(f"elapsed time is {int(minutes)} minutes and {seconds:.0f} seconds.")


        # print("Oversampling")
        # D_prime_relaxed = mechanism.get_dprime()[:, :]
        # D_prime_original_format_df = (
        #     get_syn_df(
        #         D_prime_relaxed, seed=algo_seed, oversample_rate=oversamples, domain=domain
        #     )
        # )  # Dataset object
        # print("Done")

        # if save_sycn_data:
        #     save_synthetic_dataset(
        #         mechanism_name,
        #         str(dataset_container),
        #         rho,
        #         params,
        #         algo_seed,
        #         D_prime_relaxed,
        #         D_prime_original_format_df,
        #         runtime_seconds,
        #     )


def run_experiment_old(
    mechanism: BaseMechanism,
    dataset_fn,
    rho_list: list,
    algorithm_seed: list,
    params: str,
    oversamples=40,
    save_sycn_data=True,
    run_ml_eval=False,
    get_debug_fn=None,
):
    """Runs RAP++ and saves the relaxed synthetic data as .npy."""

    if algorithm_seed is None:
        algorithm_seed = [0]
    if rho_list is None:
        rho_list = [1]

    mechanism_name = str(mechanism)

    for algo_seed in algorithm_seed:
        dataset_container = dataset_fn(algo_seed)
        true_dataset_post_df = dataset_container.from_dataset_to_df_fn(
            dataset_container.train
        )
        true_test_dataset_post_df = dataset_container.from_dataset_to_df_fn(
            dataset_container.test
        )
        cat_cols = dataset_container.cat_columns
        num_cols = dataset_container.num_columns
        labels = dataset_container.label_column
        feature_columns = list(set(cat_cols + num_cols) - set(labels))

        """ ML Test """
        # if run_ml_eval:
        #     for label in dataset_container.label_column:
        #         print(f'Original results for {label}:')
        #         evaluate_machine_learning_task(true_dataset_post_df,
        #                          true_test_dataset_post_df,
        #                         feature_columns=feature_columns,
        #                          label_column=label,
        #                          cat_columns=dataset_container.cat_columns,
        #                          endmodel=LogisticRegression(penalty='l1', solver="liblinear")
        #                          )

        debug_fn = get_debug_fn(dataset_container) if get_debug_fn is not None else None

        mechanism.initialize(
            dataset_container.train, algo_seed
        )  # Pass the datase to algorithm so that it can create the statistics.

        for rho in rho_list:
            print(
                f"\n\n\nTraining {str(mechanism)} with dataset {str(dataset_container)} and  rho={rho}, algo_seed={algo_seed}"
            )

            """ Train RAP """
            stime = time.time()
            mechanism.train(rho=rho, seed=algo_seed, debug_fn=debug_fn)
            runtime_seconds = time.time() - stime
            minutes, seconds = divmod(time.time() - stime, 60)
            print(f"elapsed time is {int(minutes)} minutes and {seconds:.0f} seconds.")

            print("Oversampling")
            D_prime_relaxed = mechanism.get_dprime()[:, :]
            D_prime_original_format_df = (
                dataset_container.get_sync_dataset_with_oversample(
                    D_prime_relaxed, seed=algo_seed, oversample_rate=oversamples
                )
            )  # Dataset object
            print("Done")

            if save_sycn_data:
                save_synthetic_dataset(
                    mechanism_name,
                    str(dataset_container),
                    rho,
                    params,
                    algo_seed,
                    D_prime_relaxed,
                    D_prime_original_format_df,
                    runtime_seconds,
                )

            """ ML Test """
            # if run_ml_eval:
            #     for label in dataset_container.label_column:
            #         print(f"Synthetic results for {label}:")
            #         evaluate_machine_learning_task(
            #             D_prime_original_format_df,
            #             true_test_dataset_post_df,
            #             feature_columns=feature_columns,
            #             label_column=label,
            #             cat_columns=dataset_container.cat_columns,
            #             endmodel=LogisticRegression(penalty="l1", solver="liblinear"),
            #         )
