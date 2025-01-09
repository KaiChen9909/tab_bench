import sys 
target_path="./"
sys.path.append(target_path)

import numpy as np
import pandas as pd
import torch
import enum
import json, pickle 
import os
import tomli, tomli_w
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, astuple, replace
from typing import Any, Callable, List, Dict, Type, Optional, Literal, Tuple, TypeVar, Union, cast, get_args, get_origin

CAT_MISSING_VALUE = '__none__'
CAT_RARE_VALUE = '__rare__'
ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']

RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')
_CONFIG_NONE = '__none__'

def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)

def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    # if config != load_config(path):
    #    raise 'Error Debug: pack/unpack'


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def load(path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](Path(path), **kwargs)


def dump(x: Any, path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'dump_{Path(path).suffix[1:]}'](x, Path(path), **kwargs)

#calculate the list of the number of categorical variables
def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]

def get_numerical_range(X: Union[torch.Tensor, np.ndarray]) -> List:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [[min(x), max(x)] for x in XT]

def get_category_range(X: Union[torch.Tensor, np.ndarray]) -> List:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist() 
    return [list(set(x)) for x in XT]

class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def get_catboost_config(real_data_path, is_cv=False):
    ds_name = Path(real_data_path).name
    C = load_json(f'eval_models/catboost/{ds_name}_cv.json')
    return C

def index_to_onehot(x, num_classes, out_type = 'array'):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x).long()
    if len(num_classes) == 1:
        x_onehot = F.one_hot(x, num_classes[0]).reshape(-1, num_classes[0])
    else: 
        onehots = []
        for i in range(len(num_classes)):
            onehots.append(F.one_hot(x[:, i], num_classes[i]))
        x_onehot = torch.cat(onehots, dim=1)
        if out_type == 'array':
            x_onehot = np.array(x_onehot)
    
    return x_onehot

def onehot_to_index(x, num_classes):
    if len(num_classes) == 1:
        decoded_indices = np.argmax(x, axis=1)
        return decoded_indices.reshape(-1, 1)
    else:
        split_indices = np.cumsum(num_classes)[:-1]
        one_hot_segments = np.split(x, split_indices, axis=1)
        decoded_indices = [np.argmax(segment, axis=1) for segment in one_hot_segments]
        return np.column_stack(decoded_indices)

def generate_shuffle_block(arr, k):
    np.random.shuffle(arr)
    subarr = [list(arr[i: i+k]) for i in range(0, len(arr), k)]

    return subarr

def round_even(x):
    y = int(np.ceil(x))
    if y % 2 == 0:
        return y
    else: 
        return y-1