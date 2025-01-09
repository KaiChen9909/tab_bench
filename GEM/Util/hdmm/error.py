# Copyright (C) 2019-2021, Tumult Labs Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from Util.hdmm.matrix import EkteloMatrix, VStack, Kronecker, Weighted
from Util.hdmm import workload

def convert_implicit(A):
    if isinstance(A, EkteloMatrix) or isinstance(A, workload.ExplicitGram):
        return A
    return EkteloMatrix(A)

def expected_error(W, A, eps=np.sqrt(2), delta=0):
    """
    Given a strategy and a privacy budget, compute the expected squared error
    """
    assert delta == 0, 'delta must be 0'
    W, A = convert_implicit(W), convert_implicit(A)
    AtA = A.gram()
    AtA1 = AtA.pinv()
    WtW = W.gram()
    if isinstance(AtA1, workload.MarginalsGram):
        WtW = workload.MarginalsGram.approximate(WtW)
    X = WtW @ AtA1
    delta = A.sensitivity()
    if isinstance(X, workload.Sum):
        trace = sum(Y.trace() for Y in X.matrices)
    else:
        trace = X.trace()
    var = 2.0 / eps**2
    return var * delta**2 * trace

def rootmse(W, A, eps=np.sqrt(2), delta=0):
    """ compute a normalized version of expected squared error """
    return np.sqrt(expected_error(W, A, eps, delta) / W.shape[0])

def squared_error(W, noise):
    """ 
    Given a noise vector (x - xhat), compute the squared error on the workload
    """
    W = convert_implicit(W)
    WtW = W.gram()
    return noise.dot(WtW.dot(noise))

def average_error_ci(W, noises):
    """
    Given a list of noise vectors (x - xhat), compute a 95% confidence interval for the mean squared error.
    """
    samples = [squared_error(W, noise) for noise in noises]
    avg = np.mean(samples)
    pm = 1.96 * np.std(samples) / np.sqrt(len(samples))
    return (avg-pm, avg+pm)

def per_query_error(W, A, eps=np.sqrt(2), delta=0, normalize=False):
    W, A = convert_implicit(W), convert_implicit(A)
    if isinstance(W, VStack):
        return np.concatenate([per_query_error(Q, A, eps, delta, normalize) for Q in W.matrices])
    delta = A.sensitivity()
    var = 2.0/eps**2
    AtA1 = A.gram().pinv()
    X = W @ AtA1 @ W.T
    err = X.diag()
    answer = var * delta**2 * err
    return np.sqrt(answer) if normalize else answer

def per_query_error_sampling(W, A, number=100000, eps=np.sqrt(2), normalize=False):
    # note: this only works for Kronecker or explicit strategy
    W, A = convert_implicit(W), convert_implicit(A)
    if isinstance(W, Weighted):
        ans = W.weight**2 * per_query_error_sampling(W.base, A, number)
    #elif isinstance(W, VStack) and type(A) == VStack:
    #    m = W.shape[0]
    #    num = lambda Wi: int(number*Wi.shape[0]/m + 1)
    #    samples = [per_query_error_sampling(Wi,Ai.base,num(Wi)) for Wi,Ai in zip(W.matrices,A.matrices)]
    #    weights = [Ai.weight for Ai in A.matrices]
    #    ans = np.concatenate([err/w**2 for w, err in zip(weights, samples)])
    elif isinstance(W, VStack):
        m = W.shape[0]
        num = lambda Wi: int(number*Wi.shape[0]/m + 1)
        samples = [per_query_error_sampling(Wi, A, num(Wi)) for Wi in W.matrices]
        ans = np.concatenate(samples)
    elif isinstance(W, Kronecker) and isinstance(A, Kronecker):
        assert isinstance(A, Kronecker)
        pieces=[per_query_error_sampling(Wi, Ai, number) for Wi,Ai in zip(W.matrices,A.matrices)]
        ans = np.prod(pieces, axis=0)
    elif isinstance(W, Kronecker) and isinstance(A, workload.Marginals):
        # optimization: if W is Marginals, all errors are the same
        if all( type(Wi) in [workload.Identity, workload.Ones] for Wi in W.matrices ):
            err = expected_error(W, A)
            ans = np.repeat(err, number)
        else:
            # will be very slow, uses for loop
            AtA1 = A.gram().pinv()
            ans = np.zeros(number)
            for i in range(number):
                idx = [np.random.randint(Wi.shape[0]) for Wi in W.matrices]
                w = Kronecker([Wi[j] for Wi, j in zip(W.matrices, idx)])
                ans[i] = expected_error(w, A)
    else:
        ans = np.random.choice(per_query_error(W, A), number)
        delta = A.sensitivity()
    ans *= 2.0/eps**2
    return np.sqrt(ans) if normalize else ans

def strategy_supports_workload(W, A):
    '''
    :param W: workload
    :param A: strategy
    :return: True is W is supported by A
    '''
    AtA = A.gram()
    AtA1 = AtA.pinv()
    WtW = W.gram()
    if isinstance(AtA1, workload.MarginalsGram):
        WtW = workload.MarginalsGram.approximate(WtW)
    X = WtW @ AtA1 @ AtA
    y = np.random.rand(WtW.shape[1])
    return np.allclose(WtW @ y, X @ y)
