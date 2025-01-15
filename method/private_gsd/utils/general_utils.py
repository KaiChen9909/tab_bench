import time
from method.private_gsd.utils import Domain
import matplotlib.pyplot as plt


def timer(last_time=None, msg=None):
    now = time.time()
    if msg is not None and last_time is not None:
        print(f'{msg} {now - last_time:.5f}')
    return now


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