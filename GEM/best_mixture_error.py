import sys
sys.path.append("../private-pgm/src")

import time
import datetime
import pickle
import argparse
from tqdm import tqdm

import Util.util_general as util
from Util.data_pub_sampling import *
from Util.qm import QueryManager

import pdb

# convert data (pd.DataFrame) into onehot encoded records
def get_data_onehot(data):
    df_data = data.df.copy()
    dim = np.sum(data.domain.shape)

    i = 0
    for attr in data.domain.attrs:
        df_data[attr] += i
        i += data.domain[attr]
    data_values = df_data.values

    data_onehot = np.zeros((len(data_values), dim))
    arange = np.arange(len(data_values))
    arange = np.tile(arange, (data_values.shape[1], 1)).T
    data_onehot[arange, data_values] = 1

    return data_onehot

def generate_nondp(data, real_answers, A_init, query_manager, early_stopping=50, return_last=False, log_interval=1):
    data_onehot = None
    A = np.copy(A_init)

    # initialize A_avg so that we can take an average of all A_t's at the end
    A_avg = np.zeros(A_init.shape)

    data_onehot = get_data_onehot(data)

    start_time = time.time()
    iteration = 0
    iters_since_improvement = 0
    best_error = np.infty
    while(True):
        iteration += 1
        if query_manager.xy is not None:
            fake_answers = query_manager.get_answer_weights(A) # saves runtime at the cost of memory
        else:
            fake_answers = query_manager.get_answer(data, weights=A)
        score = np.abs(real_answers - fake_answers)
        q_t_ind = score.argmax()

        m_t = real_answers[q_t_ind]

        # Multiplicative Weights update
        if query_manager.query_attrs is not None:
            query_attrs = query_manager.query_attrs[q_t_ind]
            query_mask = query_attrs != -1
            q_t_x = data.df.values[:, query_mask] - query_attrs[query_mask]
            q_t_x = np.abs(q_t_x).sum(axis=1)
            q_t_x = (q_t_x == 0).astype(int)
        else:
            if data_onehot is None:
                data_onehot = get_data_onehot(data)
            query = query_manager.get_query_workload([q_t_ind])
            q_t_x = data_onehot.dot(query.T).flatten()
            q_t_x = (q_t_x == query.sum()).astype(int)
        q_t_A = fake_answers[q_t_ind]

        factor = np.exp(q_t_x * (m_t - q_t_A))
        A = A * factor
        A = A / A.sum()
        A_avg += A

        error = score.max()
        if error < best_error:
            best_error = error
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1

        if iters_since_improvement > early_stopping:
            break

        if iteration % log_interval == 0:
            time_elapsed = int(time.time() - start_time)
            time_elapsed = str(datetime.timedelta(seconds=time_elapsed))
            log = "({}) elapsed: {}, iters_stop: {}, error: {:.6f}".format(iteration, time_elapsed, iters_since_improvement, best_error)
            print(log)

    A_avg /= (iteration + 1)

    if return_last:
        return A_avg, A
    return A_avg

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=100)
    # acs params
    parser.add_argument('--state', type=str, default=None)

    args = parser.parse_args()

    print(args)
    return args

if __name__ == "__main__":
    args = get_args()

    results_dir ='results/{}'.format(args.dataset)
    if args.dataset.startswith('acs_'):
        results_dir += '_{}'.format(args.state)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        if args.dataset.startswith('acs'):
            args.dataset = args.dataset[:-6]

    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter_private)
    query_manager = QueryManager(data.domain, workloads)
    N = data.df.shape[0]

    data_support, A_init = get_support(data)

    # get answers and initial error
    real_answers = query_manager.get_answer(data)
    query_manager.setup_xy(data_support)
    fake_answers = query_manager.get_answer_weights(A_init)
    init_error = np.abs(real_answers - fake_answers).max()

    A_avg, A_last = generate_nondp(data_support, real_answers, A_init, query_manager, early_stopping=args.early_stopping, return_last=True)

    fake_answers = query_manager.get_answer_weights(A_avg)
    max_error_avg = np.abs(real_answers - fake_answers).max()

    fake_answers = query_manager.get_answer_weights(A_last)
    max_error_last = np.abs(real_answers - fake_answers).max()

    # collect results
    result_cols = {'marginal': args.marginal,
                   'num_workloads': args.workload,
                   'workload_seed': args.workload_seed,
                   'num_queries': query_manager.num_queries,
                   }

    df_results = pd.DataFrame()
    df_results['init_error'] = [init_error]
    df_results['max_error'] = max_error_last
    i = df_results.shape[1]

    for key, val in result_cols.items():
        df_results[key] = val

    # rearrange columns for better presentation
    cols = list(df_results.columns[i:]) + list(df_results.columns[:i])
    df_results = df_results[cols]

    # save results
    results_path = os.path.join(results_dir, 'mw_nondp.csv')
    save_results(df_results, results_path=results_path)