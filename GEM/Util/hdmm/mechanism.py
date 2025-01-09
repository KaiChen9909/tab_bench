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
from Util.hdmm import workload, templates

def get_domain(W):
    if isinstance(W, workload.Kronecker):
        return tuple(Q.shape[1] for Q in W.matrices)
    elif isinstance(W, workload.Weighted):
        return get_domain(W.base)
    elif isinstance(W, workload.VStack):
        return get_domain(W.matrices[0])
    else:
        return W.shape[1]

class HDMM:

    def __init__(self, W, x, eps, seed=0):
        self.domain = get_domain(W)
        self.W = W
        self.x = x
        self.eps = eps
        self.prng = np.random.RandomState(seed)

    def optimize(self, restarts = 25):
        W = self.W
        if type(self.domain) is tuple: # kron or union kron workload
            ns = self.domain

            ps = [max(1, n//16) for n in ns]
            best_strat, best_loss = None, np.inf
            for _ in range(restarts):
                kron = templates.KronPIdentity(ps, ns)
                lossk = kron.optimize(W)
                if lossk < best_loss:
                    best_loss = lossk
                    best_strat = kron

                marg = templates.Marginals(ns)
                lossm = marg.optimize(W)
                if lossm < best_loss:
                    best_loss = lossm
                    best_strat = marg


            # multiplicative factor puts losses on same scale
            self.strategy = best_strat.strategy()
        else:
            n = self.domain
            best_strat, best_loss = None, np.inf
            for _ in range(restarts):
                pid = templates.PIdentity(max(1, n//16), n)
                loss = pid.optimize(W)
                if loss < best_loss:
                    best_loss = loss
                    best_strat = pid
            self.strategy = best_strat.strategy()
           
    def run(self):
        A = self.strategy
        A1 = A.pinv()
        delta = self.strategy.sensitivity()
        noise = self.prng.laplace(loc=0.0, scale=delta/self.eps, size=A.shape[0])
        self.ans = A.dot(self.x) + noise
        self.xest = A1.dot(self.ans)
        return self.xest 
 
