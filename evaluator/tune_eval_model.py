import sys 
target_path="./"
sys.path.append(target_path)

import optuna
import argparse
import os
from evaluator.eval_catboost import train_catboost
from evaluator.eval_mlp import train_mlp
from evaluator.eval_transformer import train_transformer
from evaluator.eval_simple import train_simple
from pathlib import Path
from evaluator.data.data_utils import * 

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('model', type=str)
parser.add_argument('tune_type', type=str)
parser.add_argument('device', type=str)

args = parser.parse_args()

os.makedirs(f'eval_models/{args.model}', exist_ok=True)
data_path = Path(f"data/{args.ds_name}")
best_params = None
info = load_json(os.path.join(data_path, 'info.json'))
task_type = info['task_type'] 

assert args.tune_type in ("cv", "val")
assert args.model in ('catboost', 'mlp', 'transformer', 'lr', 'tree', 'rf', 'mlpreg', 'svm', 'xgb')


def _suggest(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    return getattr(trial, f'suggest_{distribution}')(label, *args)

def _suggest_optional(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    if trial.suggest_categorical(f"optional_{label}", [True, False]):
        return _suggest(trial, distribution, label, *args)
    else:
        return 0.0

#
# The followings are the parameter function for different eval models
#
 
def _suggest_mlp_layers(trial: optuna.trial.Trial, mlp_d_layers: list[int]):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t

    min_n_layers, max_n_layers = mlp_d_layers[0], mlp_d_layers[1]
    d_min, d_max = mlp_d_layers[2], mlp_d_layers[3]

    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last

    return d_layers

def suggest_mlp_params(trial):
    params = {}
    params["lr"] = trial.suggest_loguniform("lr", 5e-5, 0.005)
    params["dropout"] = _suggest_optional(trial, "uniform", "dropout", 0.0, 0.5)
    params["weight_decay"] = _suggest_optional(trial, "loguniform", "weight_decay", 1e-6, 1e-2)
    params["d_layers"] = _suggest_mlp_layers(trial, [1, 8, 6, 10])

    return params

def suggest_catboost_params(trial):
    params = {}
    params["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.001, 1.0)
    params["depth"] = trial.suggest_int("depth", 3, 10)
    params["l2_leaf_reg"] = trial.suggest_uniform("l2_leaf_reg", 0.1, 10.0)
    params["bagging_temperature"] = trial.suggest_uniform("bagging_temperature", 0.0, 1.0)
    params["leaf_estimation_iterations"] = trial.suggest_int("leaf_estimation_iterations", 1, 10)

    params = params | {
        "iterations": 2000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU", # "GPU", may affect performance
        "thread_count": 4,
        # "devices": "0", # for GPU
    }

    return params 

def suggest_simple_params(trial, model_name):
    params = {}

    if model_name == 'tree':
        params['max_depth'] = trial.suggest_int("max_depth", 4, 64)
        params['min_samples_split'] = trial.suggest_int("min_samples_split", 2, 8)
        params['min_samples_leaf'] = trial.suggest_int("min_samples_leaf", 1, 8) 

    elif model_name == 'lr':
        if task_type == 'regression':
            params['alpha'] = trial.suggest_float("alpha", 0, 10)
            params['fit_intercept'] = trial.suggest_categorical("fit_intercept", [True, False])
        else: 
            params['max_iter'] = trial.suggest_int("max_iter", 100, 1000)
            params['C'] = trial.suggest_float("C", 1e-5, 1e-1, log=True)
            params['tol'] = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
    
    elif model_name == 'rf':
        params['n_estimators'] = trial.suggest_int("n_estimators", 10, 200)
        params['max_depth'] = trial.suggest_int("max_depth", 4, 64)
        params['min_samples_split'] = trial.suggest_int("min_samples_split", 2, 8)
        params['min_samples_leaf'] = trial.suggest_int("min_samples_leaf", 1, 8)

    elif model_name == 'mlpreg':
        params['max_iter'] = trial.suggest_int("max_iter", 50, 200)
        params['alpha'] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)

    elif model_name == 'svm':
        if task_type == 'regression':
            params['C'] = trial.suggest_float("C", 1e-5, 1e-1, log=True)
            params['epsilon'] = trial.suggest_float("epsilon", 1e-5, 1e-1, log=True)
        else: 
            params['C'] = trial.suggest_float("C", 1e-5, 1e-1, log=True)
            params['kernel'] = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            params['gamma'] = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    elif model_name == 'xgb':
        params['eta'] = trial.suggest_float("eta", 0.01, 0.2, log=False)
        params['min_child_weight'] = trial.suggest_int("min_child_weight", 1, 10)
        params['max_depth'] = trial.suggest_int("max_depth", 3, 20)
        params['gamma'] = trial.suggest_float("gamma", 0, 1)
    
    return params


#
# trial function for optimziation
#

def objective(trial):

# model decision
    if args.model == "mlp":
        params = suggest_mlp_params(trial)
        train_func = train_mlp
        T_dict = {
            "seed": 0,
            "normalization": "quantile",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": "one-hot",
            "y_policy": "default"
        }

    elif args.model == 'transformer':
        params = suggest_mlp_params(trial)
        train_func = train_transformer
        T_dict = {
            "seed": 0,
            "normalization": "quantile",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": "one-hot",
            "y_policy": "default"
        }

    elif args.model == 'catboost':
        params = suggest_catboost_params(trial)
        train_func = train_catboost
        T_dict = {
            "seed": 0,
            "normalization": None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        }
        
    else: # simple model
        params = suggest_simple_params(trial, args.model)
        train_func = train_simple
        T_dict = {
            "seed": 0,
            "normalization": 'minmax',
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        }
    
# train step
    trial.set_user_attr("params", params)
    if args.tune_type == "cv":
        score = 0.0
        for fold in range(5):
            metrics_report = train_func(
                parent_dir=None,
                data_path=data_path,
                eval_type="real",
                T_dict=T_dict,
                model_name = args.model,
                params=params,
                change_val=True,
                device=args.device,
                seed = fold
            )
            score += metrics_report.get_val_score()
        score /= 5

    elif args.tune_type == "val":
        metrics_report = train_func(
            parent_dir=None,
            data_path=data_path,
            eval_type="real",
            T_dict=T_dict,
            model_name = args.model,
            params=params,
            change_val=False,
            device=args.device
        )
        score = metrics_report.get_val_score()
    
    return score


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0)
)

study.optimize(objective, n_trials=10, show_progress_bar=True)
    
bets_params = study.best_trial.user_attrs['params']


best_params_path = f"eval_models/{args.model}/{args.ds_name}_{args.tune_type}.json"

dump_json(bets_params, best_params_path)