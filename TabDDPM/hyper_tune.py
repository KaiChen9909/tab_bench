import sys
target_path="./"
sys.path.append(target_path)
import subprocess
import os
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path
from TabDDPM.data.data_utils import *
from TabDDPM.scripts.pretrain_and_finetune import * 
from evaluator.eval_simple import *
from TabDDPM.scripts.sample import *

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('device', type = str, default = 'cuda:0')
parser.add_argument('--eval_model', type = str, default = 'mlpreg') # can be rf, tree, xgboost etc.

args = parser.parse_args()
ds_name = args.ds_name
device = args.device

base_config_path = f'TabDDPM/exp/{ds_name}/config.toml'
exps_path = Path(f'TabDDPM/exp/{ds_name}/many-exps/') 
os.makedirs(exps_path, exist_ok = True)


def _suggest_mlp_layers(trial, min_n_layers, max_n_layers, d_min, d_max):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    
    if args.ds_name == 'colorado':
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [2048, 4096])
        d_layers = _suggest_mlp_layers(trial, 1, 3, 8, 10) 
        steps = trial.suggest_categorical('steps', [30, 50])
    elif args.ds_name == 'loan':
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024])
        d_layers = _suggest_mlp_layers(trial, 1, 3, 8, 10) 
        steps = trial.suggest_categorical('steps', [100, 200])
    else:
        lr = trial.suggest_loguniform('lr', 0.0001, 0.005)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        d_layers = _suggest_mlp_layers(trial, 1, 4, 7, 10) 
        steps = trial.suggest_categorical('steps', [200, 400])

    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])

    base_config = load_config(base_config_path)
    dataset = make_dataset(
        base_config['real_data_path'],
        T = Transformations(**base_config['train']['T']),
        num_classes = base_config['model_params']['num_classes'],
        is_y_cond = base_config['model_params']['is_y_cond'],
        change_val = False,
        have_pretrain = 0,
        y_num_classes = base_config['model_params']['num_classes']
    )

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = 0.0
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['sample']['num_samples'] = len(dataset.y['train'])
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    base_config['eval']['T']['normalization'] = "minmax"
    base_config['eval']['T']['cat_encoding'] = None

    trial.set_user_attr("config", base_config)

    dump_config(base_config, exps_path / 'config.toml')
    os.makedirs(base_config['parent_dir'], exist_ok = True)

    finetune(
        **base_config['train']['main'],
        **base_config['diffusion_params'],
        parent_dir=base_config['parent_dir'],
        dataset = dataset,
        model_type=base_config['model_type'],
        model_params=base_config['model_params'],
        model_path = None,
        T_dict=base_config['train']['T'],
        num_numerical_features=base_config['num_numerical_features'],
        device=device,
        dp_epsilon = None,
        dp_delta = None
    )

    n_datasets = 3
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed

        sample(
            num_samples=base_config['sample']['num_samples'],
            batch_size=base_config['sample']['batch_size'],
            disbalance=base_config['sample'].get('disbalance', None),
            **base_config['diffusion_params'],
            parent_dir=base_config['parent_dir'],
            dataset = dataset,
            data_path = base_config['real_data_path'],
            model_path=os.path.join(base_config['parent_dir'], f'model.pt'),
            model_type=base_config['model_type'],
            model_params=base_config['model_params'],
            T_dict=base_config['train']['T'],
            num_numerical_features=base_config['num_numerical_features'],
            device=device,
            seed=base_config['sample'].get('seed', 0),
            dp = False
        )

        train_simple(
                parent_dir=base_config['parent_dir'],
                data_path=base_config['real_data_path'],
                eval_type=base_config['eval']['type']['eval_type'],
                T_dict=base_config['eval']['T'],
                model_name = args.eval_model,
                seed=base_config['seed'],
                change_val=True,
                device=device
            )

        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        report = load_json(report_path)

        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            score += report['metrics']['val']['macro avg']['f1-score']

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=10, show_progress_bar=True)

best_config = study.best_trial.user_attrs['config']
base_config = load_config(base_config_path)
base_config['model_params']['rtdl_params']['d_layers'] = best_config['model_params']['rtdl_params']['d_layers']
base_config['diffusion_params']['num_timesteps'] = best_config['diffusion_params']['num_timesteps']
# base_config['pretrain']['main'] = best_config['train']['main']
# base_config['train']['main'] = best_config['train']['main']
base_config['sample']['num_samples'] = best_config['sample']['num_samples'] 

dump_config(base_config, base_config_path)
