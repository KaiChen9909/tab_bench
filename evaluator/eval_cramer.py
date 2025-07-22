import numpy as np
import itertools
from collections import Counter
from evaluator.data.dataset import read_pure_data
import math

def num_divide(X_num_real, X_num_fake):
    for i in range(X_num_fake.shape[1]):
        max_value = max(X_num_real[:, i].max(), X_num_fake[:, i].max())
        min_value = min(X_num_real[:, i].min(), X_num_fake[:, i].min())
        rng = max_value - min_value or 1
        X_num_real[:, i] = np.round(99 * (X_num_real[:, i] - min_value) / rng)
        X_num_fake[:, i] = np.round(99 * (X_num_fake[:, i] - min_value) / rng)
    return X_num_real.astype(int), X_num_fake.astype(int)

def cramers_v(data, cols):
    x = data[:, cols[0]]
    y = data[:, cols[1]]
    n = len(x)
    pairs = Counter(zip(x, y))
    row_counts = Counter(x)
    col_counts = Counter(y)
    chi2 = 0.0
    for (xi, yi), o in pairs.items():
        e = row_counts[xi] * col_counts[yi] / n
        if e > 0:
            chi2 += (o - e) ** 2 / e
    r = len(row_counts)
    c = len(col_counts)
    return math.sqrt(chi2 / (n * min(r - 1, c - 1))) if min(r - 1, c - 1) > 0 else 0.0


def cramers_v_main(X_num_real, X_cat_real, y_real, X_num_fake, X_cat_fake, y_fake):
    if X_num_real is not None:
        X_num_real, X_num_fake = num_divide(X_num_real.copy(), X_num_fake.copy())
    def to_str_matrix(X_num, X_cat, y):
        parts = []
        if X_num is not None:
            parts.append(X_num.astype(str))
        if X_cat is not None:
            parts.append(X_cat.astype(str))
        parts.append(y.reshape(-1, 1).astype(str))
        return np.concatenate(parts, axis=1)
    real = to_str_matrix(X_num_real, X_cat_real, y_real)
    fake = to_str_matrix(X_num_fake, X_cat_fake, y_fake)
    p = real.shape[1]
    diffs = []
    for i, j in itertools.combinations(range(p), 2):
        v1 = cramers_v(real, (i, j))
        v2 = cramers_v(fake, (i, j))
        diffs.append(abs(v1 - v2))

    mean_diff = np.mean(diffs) if diffs else 0.0
    print(f"finish 2-way Cramer's V evaluation, mean diff = {mean_diff:.4f}")
    return {'2way CramerV diff': mean_diff}


def make_cramer(
    synthetic_data_path,
    data_path
):  
    print('-' * 100)
    print('Starting Cramer evaluation')
    X_num_real, X_cat_real, y_real = read_pure_data(data_path, split = 'test')
    X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path, split = 'train') 
    
    return cramers_v_main(
        X_num_real, X_cat_real, y_real,
        X_num_fake, X_cat_fake, y_fake
    )
