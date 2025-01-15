import os
import math
import operator
import numpy as np
import itertools
from tqdm import tqdm
from collections.abc import Iterable
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz

import sys
sys.path.append("../private-pgm/src")

import pdb

def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge

_range = range
def get_xy(sample, bins):
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # normalize the range argument
    range = (None,) * D

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                    "`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                        .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    return xy, nbin

def convert_query(query, domain):
    query = query.copy()
    query_attrs = []
    for size in domain.shape:
        if len(query) > 0 and query[0] < size:
            attr, query = query[0], query[1:]
            query_attrs.append(attr)
        else:
            query_attrs.append(-1)
        query -= size
    query_attrs = np.array(query_attrs)

    # domain_values = np.array(list(domain.config.values()))
    # domain_values_cumsum = np.cumsum(domain_values)
    # x = query_orig[:, np.newaxis] - domain_values_cumsum[np.newaxis, :] + domain_values
    # mask = (x < domain_values) & (x >= 0)
    # x[~mask] = -1
    # x = x.max(axis=0)
    # assert((query_attrs == x).mean() == 1)

    return query_attrs

def histogramdd(xy, nbin, weights):
    hist = np.bincount(xy, weights, minlength=nbin.prod())

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting='safe')

    # Remove outliers (indices 0 and -1 for each dimension).
    D = nbin.shape[0]
    core = D * (slice(1, -1),)
    hist = hist[core]

    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")
    return hist

def min_int_dtype(arr):
    max_val_abs = np.abs(arr).max()
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        if max_val_abs < np.iinfo(dtype).max:
            return dtype

def add_row_convert_dtype(array, row, idx):
    max_val_abs = np.abs(row).max()
    if max_val_abs > np.iinfo(array.dtype).max:
        dtype = min_int_dtype(row)
        array = array.astype(dtype)
    array[idx, :len(row)] = row
    return array

def get_num_queries(domain, workloads):
    col_map = {}
    for i, col in enumerate(domain.attrs):
        col_map[col] = i
    feat_pos = []
    cur = 0
    for f, sz in enumerate(domain.shape):
        feat_pos.append(list(range(cur, cur + sz)))
        cur += sz

    num_queries = 0
    for feat in workloads:
        positions = []
        for col in feat:
            i = col_map[col]
            positions.append(feat_pos[i])
        x = list(itertools.product(*positions))
        num_queries += len(x)
    return num_queries

class QueryManager():
    """< 1e-9
    K-marginal queries manager
    """
    def __init__(self, domain, workloads):
        self.domain = domain
        self.workloads = workloads
        self.att_id = {}
        col_map = {}
        for i,col in enumerate(self.domain.attrs):
            col_map[col] = i
            self.att_id[col] = i
        feat_pos = []
        cur = 0
        for f, sz in enumerate(domain.shape):
            feat_pos.append(list(range(cur, cur + sz)))
            cur += sz
        self.dim = np.sum(self.domain.shape)
        self.num_queries = get_num_queries(self.domain, self.workloads)
        self.max_marginal = np.array([len(x) for x in self.workloads]).max()

        dtype = min_int_dtype([self.dim])
        self.queries = -1 * np.ones((self.num_queries, self.max_marginal), dtype=dtype) # (query num, marginal dim)
        idx = 0
        print("Initializing self.queries...")
        for feat in tqdm(self.workloads):
            positions = []
            for col in feat:
                i = col_map[col]
                positions.append(feat_pos[i])
            x = list(itertools.product(*positions))
            x = np.array(x)
            self.queries[idx:idx+x.shape[0], :x.shape[1]] = x
            idx += x.shape[0]

        self.feat_pos = feat_pos
        self.xy = None
        self.nbin = None
        self.query_attrs = None
        self.q_x = None

    def get_small_separator_workload(self):
        W = []
        for i in range(self.dim):
            w = np.zeros(self.dim)
            w[i] = 1
            W.append(w)
        return np.array(W)

    def get_query_workload(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        W = []
        for q_id in q_ids:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                if p < 0:
                    break
                w[p] = 1
            W.append(w)
        if len(W) == 1:
            W = np.array(W).reshape(1, -1)
        else:
            W = np.array(W)
        return W

    def get_query_workload_weighted(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        wei = {}
        for q_id in q_ids:
            wei[q_id] = 1 + wei[q_id] if q_id in wei else 1
        W = []
        weights = []
        for q_id in wei:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                w[p] = 1
            W.append(w)
            weights.append(wei[q_id])
        if len(W) == 1:
            W = np.array(W).reshape(1,-1)
            weights = np.array(weights)
        else:
            W = np.array(W)
            weights = np.array(weights)
        return W, weights

    def get_answer(self, data, weights=None, concat=True, debug=False):
        ans_vec = []
        N_sync = data.df.shape[0]
        # for proj, W in self.workloads:
        for proj in self.workloads:
            # weights let's you do a weighted sum
            x = data.project(proj).datavector(weights=weights)
            if weights is None:
                x = x / N_sync
            ans_vec.append(x)
        if concat:
            ans_vec = np.concatenate(ans_vec)
        return ans_vec

    def setup_query_workload(self):
        domain_values = np.array(list(self.domain.config.values()))
        domain_values_cumsum = np.cumsum(domain_values)
        domain_values = domain_values.astype(min_int_dtype(domain_values))
        domain_values_cumsum = domain_values_cumsum.astype(min_int_dtype(domain_values_cumsum))

        shape = (len(self.queries), len(domain_values))
        self.query_attrs = -1 * np.ones(shape, dtype=np.int8)

        idx = 0
        num_chunks = math.ceil(shape[0] / int(1e7))
        for queries in tqdm(np.array_split(self.queries, num_chunks)):
            x = queries[:, :, np.newaxis] - domain_values_cumsum[np.newaxis, np.newaxis, :] + domain_values
            mask = (x < domain_values) & (x >= 0)
            x[~mask] = -1
            x = x.max(axis=1)

            dtype = min_int_dtype(x)
            self.query_attrs = self.query_attrs.astype(dtype, copy=False)
            self.query_attrs[idx:idx+x.shape[0]] = x
            idx += x.shape[0]

    def setup_query_attr(self, save_dir=None):
        path = None
        if save_dir is not None:
            path = os.path.join(save_dir, 'query_attr.npy')
            if os.path.exists(path):
                self.query_attrs = np.load(path)
                self.queries = None
                return

        print('running init_query_attr...')
        self.setup_query_workload()

        if path is not None:
            np.save(path, self.query_attrs)
        # when using init_query_attr, I don't think you will need self.queries anymore. Change this later if we run into a situation where it's still needed
        self.queries = None

    def setup_xy(self, data, save_dir=None, overwrite=False):
        path_xy, path_nbin = None, None
        if save_dir is not None:
            path_xy = os.path.join(save_dir, 'xy.npy')
            path_nbin = os.path.join(save_dir, 'nbin.npy')
            if not overwrite and os.path.exists(path_xy) and os.path.exists(path_nbin):
                self.xy = np.load(path_xy)
                self.nbin = np.load(path_nbin)
                return

        print('running set_up_xy...')
        self.xy = None
        self.nbin = None
        for i, proj in enumerate(tqdm(self.workloads)):
            _data = data.project(proj)
            bins = [range(n + 1) for n in _data.domain.shape]
            xy, nbin = get_xy(_data.df.values, bins)

            if self.xy is None:
                shape = (len(self.workloads), xy.shape[0])
                self.xy = -1 * np.ones(shape, dtype=np.int8)
            if self.nbin is None:
                shape = (len(self.workloads), nbin.shape[0])
                self.nbin = -1 * np.ones(shape, dtype=np.int8)

            self.xy = add_row_convert_dtype(self.xy, xy, i)
            self.nbin = add_row_convert_dtype(self.nbin, nbin, i)

        if path_xy is not None and path_nbin is not None:
            np.save(path_xy, self.xy)
            np.save(path_nbin, self.nbin)

    def get_answer_weights(self, weights, concat=True):
        assert(self.xy is not None) # otherwise set_up_xy hasn't been run
        ans_vec = []
        xy_neg = -1 in self.xy
        nbin_neg = -1 in self.nbin
        for i in range(len(self.workloads)):
            xy = self.xy[i]
            nbin = self.nbin[i]
            if xy_neg:
                xy = xy[xy != -1]
            if nbin_neg:
                nbin = nbin[nbin != -1]
            x = histogramdd(xy, nbin, weights).flatten()
            ans_vec.append(x)
        if concat:
            ans_vec = np.concatenate(ans_vec)
        return ans_vec

    # doesn't speed things
    def setup_q_x(self, data, save_dir=None):
        path = None
        if save_dir is not None:
            path = os.path.join(save_dir, 'q_x.npz')
            if os.path.exists(path):
                self.q_x = load_npz(path)
                return

        self.q_x = []
        GB_TOTAL = 3
        gb = 1e-9 * data.df.shape[0]
        chunk_size = int(GB_TOTAL / gb)
        num_chunks = math.ceil(len(self.query_attrs) / chunk_size)
        for query_attrs_chunk in np.array_split(self.query_attrs, num_chunks):
            shape = (query_attrs_chunk.shape[0], data.df.shape[0])
            q_x = np.zeros(shape, dtype=np.byte)
            for idx, query_attrs in enumerate(tqdm(query_attrs_chunk)):
                query_mask = query_attrs != -1
                q_t_x = data.df.values[:, query_mask] - query_attrs[query_mask]
                q_t_x = np.abs(q_t_x).sum(axis=1)
                q_t_x = (q_t_x == 0).astype(np.byte)
                q_x[idx] = q_t_x
            q_x = csr_matrix(q_x, dtype=np.byte)
            self.q_x.append(q_x)
        self.q_x = vstack(self.q_x)

        if path is not None:
            save_npz(path, self.q_x)





















