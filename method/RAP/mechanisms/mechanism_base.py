from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp

from method.RAP.dataloading.dataset import Dataset


@dataclass
class BaseConfiguration:
    # Logging
    verbose: bool = False
    silent: bool = False
    debug: bool = False
    privacy_weight: int = 1


class BaseMechanism(metaclass=ABCMeta):
    def __init__(
        self,
        args: list,
        stats_module: list,
        num_generated_points: int = 1000,
        name="Base",
    ):
        self.args_list = args
        self.stat_module = stats_module
        self.num_generated_points = num_generated_points
        self.algo_name = name

        self.dataset = None
        self.statistics = None
        self.privacy_weights = None
        self.D_prime = None
        self.num_points = 0
        self.data_dimension = None

    def __str__(self):
        return self.algo_name

    def initialize(self, dataset: Dataset, seed: int):
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        def print_gpu_memory_pynvml(message=""):
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)  
            info = nvmlDeviceGetMemoryInfo(handle)
            print(f"{message} Memory Used: {info.used / 1024**3:.2f} GB, Total: {info.total / 1024**3:.2f} GB")

        domain = dataset.domain
        self.dataset = dataset
        self.num_points = len(self.dataset.df)
        self.data_dimension = sum(domain.shape)
        self.statistics = [
            stat.get_statistics(domain, seed) for stat in self.stat_module
        ]
    

    def get_dprime(self):
        return self.D_prime

    def train(self, rho, seed, debug_fn=None, num_generated_points=None): 
        if num_generated_points is not None:
            self.num_generated_points = num_generated_points

        if self.dataset is None or self.statistics is None:
            raise Exception("Error must call initialize()")

        self._train(self.args_list, self.num_generated_points, rho, seed, debug_fn)

    @abstractmethod
    def _train(
        self, args: list, num_generated_points: int, rho: float, seed: int, debug_fn
    ):
        pass

    def _clip_array(self, array):
        return jnp.clip(array, 0, 1)
