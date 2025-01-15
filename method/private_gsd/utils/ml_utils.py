import jax.random
import matplotlib.pyplot as plt
import pandas as pd
from method.private_gsd.utils import Domain
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def filter_outliers(df_train, df_test, config, quantile=0.01, visualize_columns=False):
    domain = Domain.fromdict(config)

    df = df_train.append(df_test)

    for num_col in domain.get_numeric_cols():
        q_lo = df[num_col].quantile(quantile)
        q_hi = df[num_col].quantile(1-quantile)

        df_train.loc[df_train[num_col] > q_hi, num_col] = q_hi
        df_test.loc[df_test[num_col] > q_hi, num_col] = q_hi

        df_train.loc[df_train[num_col] < q_lo, num_col] = q_lo
        df_test.loc[df_test[num_col] < q_lo, num_col] = q_lo

    # Rescale data
    df_outlier = df_train.append(df_test)

    for num_col in domain.get_numeric_cols():
        maxv = df_outlier[num_col].max()
        minv = df_outlier[num_col].min()

        df_train[num_col] = (df_train[num_col] - minv) / (maxv - minv)
        df_test[num_col] = (df_test[num_col] - minv) / (maxv - minv)
        # meanv = df_train[num_col].mean()
        # print(f'Col={num_col:<10}: mean={meanv:<5.3f}, min={minv:<5.3f},  max={maxv:<5.3f},')
        if visualize_columns:
            df_train[num_col].hist()
            plt.title(f'Column={num_col}')
            # plt.yscale('log')
            plt.show()

    return df_train, df_test

def separate_cat_and_num_cols(domain, features):
    # train_cols = [c for c in domain.attrs if c != target]
    train_cols_num = [c for c in features if domain[c] == 1]
    train_cols_cat = [c for c in features if c not in train_cols_num]
    return train_cols_num, train_cols_cat
def get_Xy(domain: Domain, features: list, target, df_train: pd.DataFrame, df_test: pd.DataFrame,
           rescale=True):
    cols_num, cols_cat = separate_cat_and_num_cols(domain, features)
    y_train = df_train[target].values
    y_test = df_test[target].values
    X_train = None
    X_test = None


    if len(cols_cat) > 0:
        X_cat_train = df_train[cols_cat].values
        X_cat_test = df_test[cols_cat].values
        X_cat_all = np.vstack((X_cat_train, X_cat_test))
        categories = [np.arange(domain[c]) for c in cols_cat]
        enc = OneHotEncoder(categories=categories)
        enc.fit(X_cat_all)
        X_cat_train = enc.transform(X_cat_train).toarray()
        X_cat_test = enc.transform(X_cat_test).toarray()
        X_train = X_cat_train
        X_test = X_cat_test

    if len(cols_num) > 0:
        X_num_train = df_train[cols_num].values
        X_num_test = df_test[cols_num].values

        if rescale:
            scaler = StandardScaler()
            scaler.fit(X_num_train)
            X_num_train = scaler.transform(X_num_train)
            X_num_test = scaler.transform(X_num_test)

        if X_train is not None:
            X_train = np.concatenate((X_train, X_num_train), axis=1)
            X_test = np.concatenate((X_test, X_num_test), axis=1)
        else:
            X_train = X_num_train
            X_test = X_num_test

    assert X_train is not None
    assert X_test is not None
    return X_train, y_train, X_test, y_test