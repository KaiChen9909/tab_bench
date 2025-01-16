from sklearn.metrics import classification_report, r2_score, f1_score
import numpy as np
import os
import sys
target_path="./"
sys.path.append(target_path)

from sklearn.utils import shuffle
import random
from evaluator.data.data_utils import *
from evaluator.data.dataset import * 
from evaluator.data.metrics import * 
from pathlib import Path
from evaluator.util import TabTransformer
from skorch.regressor import NeuralNetRegressor
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import Dataset as SkDataset
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import predefined_split
from torch.optim import AdamW
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss



def train_transformer(
    parent_dir,
    data_path, # where test data store
    eval_type,
    T_dict,
    params = None,
    change_val = False,
    seed = 0,
    device = "cuda:0",
    model_step = 'finetune',
    model_name = None
):
    random.seed(seed)
    synthetic_data_path = os.path.join(parent_dir) if parent_dir is not None else None
    info = load_json(os.path.join(data_path, 'info.json'))
    T = Transformations(**T_dict)
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(data_path, val_size=0.2, model_step=model_step, seed=seed)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data')
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
        print('loading synthetic data from:', synthetic_data_path)
        X_num, X_cat, y = read_pure_data(synthetic_data_path, 'train' if model_step=='finetune' else 'pretrain')

    elif eval_type == 'real':
        print('loading real data from', data_path)
        if not change_val:
            X_num, X_cat, y = read_pure_data(data_path, 'train' if model_step=='finetune' else 'pretrain')
        else: 
            X_num, X_cat, y = X_num_real, X_cat_real, y_real

    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(data_path, 'val')
    X_num_test, X_cat_test, y_test = read_pure_data(data_path, 'test')

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

    X["train"], D.y["train"] = shuffle(X["train"], D.y["train"], random_state=seed)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    if params is None:
        if model_step == 'finetune':
            params = load_json(f"eval_models/transformer/{Path(data_path).name}_cv.json")
        elif model_step == 'pretrain': 
            params = load_json(f"eval_models/transformer/{Path(data_path).name}_pre_cv.json")

    mlp_params = {}
    if params is not None:
        mlp_params["d_layers"] = params["d_layers"]
        mlp_params["dropout"] = params["dropout"]
        mlp_params["d_in"] = X["train"].shape[1]
        mlp_params["d_out"] = D.nn_output_dim

    model = TabTransformer.make_baseline(**mlp_params)

    if D.is_regression:
        y = {k: D.y[k].reshape(-1, 1).astype(np.float32) for k in D.y}
    elif D.is_binclass:
        y = {k: D.y[k].reshape(-1, 1).astype(np.float32) for k in D.y}
    else:
        y = {k: D.y[k].astype(np.int64) for k in D.y}

    train_ds = SkDataset(X = X["train"].to_numpy(), y = y["train"])
    val_ds = SkDataset(X = X["val"].to_numpy(), y = y["val"])
    es = EarlyStopping(monitor="valid_loss", patience=16)

    print('-'*100)

    def f1(net, X, y):
        y_pred = net.predict(X)
        return f1_score(y, y_pred, average="macro")

    def r2(net, X, y):
        y_pred = net.predict(X)
        return r2_score(y, y_pred)

    if D.is_regression:
        net = NeuralNetRegressor(
            model,
            criterion=MSELoss,
            optimizer=AdamW,
            lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            batch_size=128 if len(D.y["train"]) < 10_000 else 256,
            max_epochs=1000,
            train_split=predefined_split(val_ds),
            iterator_train__shuffle=True,
            device=device,
            callbacks=[es, EpochScoring(r2, lower_is_better=False)],
        )

    else:
        net = NeuralNetClassifier(
            model,
            criterion=BCEWithLogitsLoss if D.is_binclass else CrossEntropyLoss,
            optimizer=AdamW,
            lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            batch_size=128 if len(D.y["train"]) < 10_000 else 256,
            max_epochs=1000,
            train_split=predefined_split(val_ds),
            iterator_train__shuffle=True,
            device=device,
            callbacks=[es, EpochScoring(f1, lower_is_better=False)],
        )

    net.fit(
        X=train_ds.X,
        y=train_ds.y
    )

    print("LAST:", len(net.history))

    predictions = {k: net.predict_proba(v.to_numpy())[:, 1] if D.is_binclass else 
                      net.predict_proba(v.to_numpy()) if D.is_multiclass else 
                      net.predict(v.to_numpy()) 
        for k, v in X.items()
    }

    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        dump_json(report, os.path.join(parent_dir, "results_transformer.json" if model_step=='finetune' else 'pretrain_results_transformer.json'))

    return metrics_report