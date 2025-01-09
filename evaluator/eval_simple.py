import numpy as np
import os
import random
import sys
target_path="./"
sys.path.append(target_path)

from sklearn.utils import shuffle
from pathlib import Path
from evaluator.data.data_utils import *
from evaluator.data.dataset import * 
from evaluator.data.metrics import * 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from xgboost import XGBClassifier, XGBRegressor

def train_simple(
    parent_dir,
    data_path,
    eval_type,
    T_dict,
    model_name = "tree",
    seed = 0,
    change_val = False,
    params = None, # dummy
    device = None, # dummy
    model_step = 'finetune'
):
    np.random.seed(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)

    T_dict["normalization"] = "minmax"
    T_dict["cat_encoding"] = None
    T = Transformations(**T_dict)
    info = load_json(os.path.join(data_path, 'info.json'))
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(data_path, val_size=0.2, model_step=model_step)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(data_path, 'train' if model_step=='finetune' else 'pretrain')
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, 'train' if model_step=='finetune' else 'pretrain')

        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {parent_dir}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path, 'train' if model_step=='finetune' else 'pretrain')

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(data_path, 'train' if model_step=='finetune' else 'pretrain')
        else: 
            X_num, X_cat, y = X_num_real, X_cat_real, y_real

    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(data_path, 'val' if model_step=='finetune' else 'preval')
    X_num_test, X_cat_test, y_test = read_pure_data(data_path, 'test' if model_step=='finetune' else 'pretest')

    D = Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        TaskType(info['task_type']),
        info.get('n_classes')
    )

    D = transform_dataset(D, T, None)
    X = concat_features(D)
    # ixs = np.random.choice(len(D.y["train"]), min(info["train_size"], len(D.y["train"])), replace=False)
    # X["train"] = X["train"].iloc[ixs]
    # D.y["train"] = D.y["train"][ixs]

    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')
    print(T_dict)
    print('-'*100)

    if params is None:
        if model_step == 'finetune':
            params = load_json(f"eval_models/{model_name}/{Path(data_path).name}_cv.json")
        elif model_step == 'pretrain': 
            params = load_json(f"eval_models/{model_name}/{Path(data_path).name}_pre_cv.json")
    
    if D.is_regression:
        if model_name == 'tree':
            model = DecisionTreeRegressor(**params, random_state=seed)
        elif model_name == 'rf':
            model = RandomForestRegressor(**params, random_state=seed)
        elif model_name == 'lr':
            model = Ridge(**params, random_state=seed)
        elif model_name == 'mlpreg':
            model = MLPRegressor(**params, random_state=seed)
        elif model_name == 'svm':
            model = SVR(**params, random_state=seed) 
        elif model_name == 'xgb':
            model = XGBRegressor(**params, 
                                  objective="reg:squarederror",
                                  random_state=seed
                                )
        else:
            raise 'Please enter a valid model name'
    else:
        if model_name == 'tree':
            model = DecisionTreeClassifier(**params, random_state=seed)
        elif model_name == 'rf':
            model = RandomForestClassifier(**params, random_state=seed)
        elif model_name == 'lr':
            model = LogisticRegression(**params, n_jobs=-1, 
                                       multi_class="multinomial" if info['task_type'] == 'multiclass' else "auto", 
                                       random_state=seed
                                      )
        elif model_name == 'mlpreg':
            model = MLPClassifier(**params, random_state=seed)
        elif model_name == 'svm':
            model = SVC(**params, probability=True, random_state=seed) 
        elif model_name == 'xgb':
            model = XGBClassifier(**params, 
                                  objective="binary:logistic" if info['task_type'] == "binclass" else "multi:softmax",
                                  random_state=seed
                                )
        else:
            raise 'Please enter a valid model name'

    predict = (
        model.predict if D.is_regression
        else 
        model.predict_proba if D.is_multiclass
        else lambda x: model.predict_proba(x)[:, 1]
    )

    model.fit(X['train'], D.y['train'])

    predictions = {k: predict(v) for k, v in X.items()}

    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = MetricsReport(report['metrics'], D.task_type)
    print(model.__class__.__name__)
    metrics_report.print_metrics()
    
    if parent_dir is not None:
        dump_json(report, os.path.join(parent_dir, f"results_{model_name}.json"))

    return metrics_report

    