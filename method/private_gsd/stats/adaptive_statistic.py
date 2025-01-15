

from typing import Callable

import chex

from method.private_gsd.utils import Dataset, Domain


class AdaptiveStatisticState:
    domain: Domain

    def get_domain(self):
        return self.domain

    def get_num_workloads(self) -> int:
        pass

    def _get_workload_fn(self, workload_ids: list = None) -> Callable:
        pass

    def _get_dataset_statistics_fn(self, workload_ids: list = None, jitted: bool = False) -> Callable:
        pass

    def _get_diff_workload_fn(self, workload_ids: list = None) -> Callable:
        pass

    def _get_stat_fn(self, query_ids: chex.Array):
        pass

    def _get_workload_sensitivity(self, workload_id: int = None, N: int = None) -> float:
        pass

    def _get_workload_positions(self, workload_id: int = None) -> tuple:
        pass

