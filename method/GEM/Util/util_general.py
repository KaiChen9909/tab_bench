import numpy as np
import sys,os
import itertools
import time

from method.GEM.Util.cdp2adp import cdp_rho

def progress_bar(T, t, msg=""):
    bar_len = 20
    progress = t / T
    sys.stdout.write("[")
    for i in range(bar_len):
        c = "-" if i < bar_len*progress else " "
        sys.stdout.write(c)
    sys.stdout.write("] {:.4f} out of {:.4f}: {}\r".format(t, T, msg))
    sys.stdout.flush()

def get_k_marginal_queries(dim, K):
    assert K<dim
    queries = list(itertools.combinations(np.arange(0, dim), K))
    return queries

def get_error(Q_samp, D_samp):
    if len(Q_samp)==0:return 0
    error = 0
    for q in Q_samp:
        error += q.eval(D_samp)
    return error/len(Q_samp)

def get_expected_payoff(query_manager, Q_dist, D_hat):
    expected_payoff = 0
    for q_i in range(query_manager.num_queries):
        A =  query_manager.payoff(q_i, D_hat)
        expected_payoff += Q_dist[q_i] * A
    return expected_payoff

def privacy_spent_adv_comp(round_eps, delta):
    term1 = 0
    sum_eps2 = 0
    for e_t in round_eps:
        term1 += e_t*(np.exp(e_t)-1)
        sum_eps2 += e_t**2
    term2 = np.sqrt(sum_eps2*np.log(1/delta)/2)
    # print("\nDebug adv comp:\n")
    # print("round_eps = ", round_eps)
    # print("term2 = {}".format(term2))
    return term1 + term2

def get_em_dist(score):
    """
    Return the distribution of the exponential mechanism
    while checking for errors
    """
    EM_dist_0 = np.exp(score, dtype=np.float128)
    sum = np.sum(EM_dist_0) # normalize EM_distribution
    assert not np.isnan(EM_dist_0).any(), "EM_dist = {}, sum = {}".format(EM_dist_0, sum)
    assert sum > 0
    assert not np.isinf(sum)
    EM_dist = EM_dist_0/sum
    assert not np.isnan(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)
    assert not np.isinf(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)
    return EM_dist

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def create_dir(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def decode_dataset(oh_data, domain):
    if type(oh_data) is list:
        # print("Debug 1")
        # print("oh_data = ", oh_data)
        oh_data= np.array(oh_data)
        # print("oh_data = ", oh_data)
    if len(oh_data.shape) == 1:
        # import pdb
        # pdb.set_trace()
        # print("Debug 2")
        # print("oh_data = ", oh_data)
        oh_data = oh_data.reshape(1,-1)
        # print("oh_data = ", oh_data)

    assert oh_data.shape[0]>0, "{}".format(oh_data.shape[0])
    assert oh_data.shape[1]>0, "{}".format(oh_data.shape[1])
    data = []
    dim = len(domain.shape)
    for oh_row in oh_data:
        assert np.sum(oh_row) == dim, "np.sum(oh_row) = {}\t{}".format(int(np.sum(oh_row)), len(domain.shape))
        c = 0
        row = []
        for att_size in domain.shape:
            feat = np.array(oh_row[c:c+att_size])
            assert np.sum(feat) == 1, "np.sum(feat) = {}".format(np.sum(feat))
            # print("feat = ", feat)
            # print("np.where(feat == 1)", np.where(feat == 1))
            val = np.where(feat == 1)[0][0]
            row.append(val)
            c+= att_size
        data.append(row)
    return np.array(data)

def get_rounds_zCDP(epsilon, eps0, adaptive, delta):
    assert adaptive > 0 and adaptive < 1
    def from_dp_to_zcdp(e):
        return 0.5 * e ** 2
    t = 1
    rho = 0
    while True:
        eps_t = eps0 + adaptive * (t-1)  # get eps for round T
        rho = rho + from_dp_to_zcdp(eps_t)  # composition
        total_epsilon = rho + 2 * np.sqrt(rho * np.log(1 / delta))
        if total_epsilon > epsilon: break
        t += 1
    return t - 1

########

def get_errors(real_answers, fake_answers):
    errors = []
    squared_errors = []
    for workload_idx in range(len(real_answers)):
        workload_errors = real_answers[workload_idx] - fake_answers[workload_idx]
        errors.append(workload_errors)

        squared_error = np.linalg.norm(workload_errors, ord=2) ** 2 / len(workload_errors)
        squared_errors.append(squared_error)
    errors = np.abs(np.concatenate(errors))

    results = {'max': np.max(errors),
               'mean': np.mean(errors),
               'median': np.median(errors),
               'mean_squared': np.linalg.norm(errors, ord=2) ** 2 / len(errors),
               'mean_workload_squared': np.mean(squared_errors),
               }
    return results

def sample(dist):
    cumulative_dist = np.cumsum(dist)
    r = np.random.rand()
    return np.searchsorted(cumulative_dist, r)

def sample_permute(dist):
    permutation = np.random.permutation(len(dist))
    dist_permuted = dist[permutation]
    ind_permuted = sample(dist_permuted)
    ind = permutation[ind_permuted]
    return ind

def exponential_mechanism(score, eps0, N):
    EM_dist_0 = np.exp(eps0 * N * score, dtype=np.float128)  # Note: sensitivity is 1/N
    EM_dist = EM_dist_0 / EM_dist_0.sum()
    query_t_index = sample(EM_dist)
    return query_t_index

def get_eps0_zCDP(epsilon, delta, T, alpha=0.5):
    rho = cdp_rho(epsilon, delta)
    eps0 = (2 * rho) / (T * (alpha ** 2 + (1 - alpha) ** 2))
    eps0 = eps0 ** 0.5
    return eps0, rho

def get_eps0_simple(rho, T, alpha=0.5):
    eps0 = (2 * rho) / (T * (alpha ** 2 + (1 - alpha) ** 2))
    eps0 = eps0 ** 0.5
    return eps0

def get_rounds(epsilon, eps0, delta):
    A = eps0*(np.exp(eps0)-1)
    B = np.sqrt(2*np.log(1/delta))*eps0
    C = -epsilon

    sqrtT_1 = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    sqrtT_2 = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    if sqrtT_1 > 0:
        T = sqrtT_1**2
    else:
        T = sqrtT_2**2
    assert np.abs(epsilon - (np.sqrt(2*T*np.log(1/delta))*eps0 + T*eps0*(np.exp(eps0)-1))) < 1e-7, 'eps0 = {}, T = {}'.format(eps0, T)
    return int(T)

def get_eps0(T, epsilon, delta):
    _T = None
    start = 0
    end = epsilon
    while _T != T:
        mid = (start + end) / 2
        _T = get_rounds(epsilon, mid, delta)
        if _T > T:
            start = mid
        else:
            end = mid
    return mid

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