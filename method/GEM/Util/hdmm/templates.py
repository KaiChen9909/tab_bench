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


from Util.hdmm import matrix, workload
from functools import reduce
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg.lapack import dpotrf, dpotri

class TemplateStrategy:

    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(2**32-1)

        self.prng = np.random.RandomState(seed)
        
    def strategy(self):
        pass
  
    def _AtA1(self):
        return self.strategy().gram().pinv().dense_matrix()
 
    def _loss_and_grad(self, params):
        pass

    def _set_workload(self, W):
        self._workload = W
        self._gram = W.gram()

    def optimize(self, W, init=None):
        """
        Optimize strategy for given workload 
        :param W: the workload, may be a n x n numpy array for WtW or a workload object
        """
        self._set_workload(W)
        if init is None:
            init = self.prng.rand(self._params.size)
        bnds = [(0,None)]*init.size
       
        opts = { 'ftol' : 1e-4 }
        res = optimize.minimize(self._loss_and_grad, init, jac=True, method='L-BFGS-B', bounds=bnds, options=opts)
        self._params = np.maximum(0, res.x)
        return res.fun       
 
class BestTemplate(TemplateStrategy):
    """
    Optimize strategy using several templates and give the best one
    """
    def __init__(self, templates, seed=None):
        super(BestTemplate, self).__init__(seed)

        self.templates = templates
        self.best = self.templates[0]

    def strategy(self):
        return self.best.strategy()

    def optimize(self, W):
        if isinstance(W, workload.Ones):
            self.best = Total(W.shape[1])
            return self.best.optimize(W) 
        best_loss = np.inf
        losses = []
        for temp in self.templates:
            loss = temp.optimize(W)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                self.best = temp
        self._errors = np.array(losses)
        #print(best_loss, self.best)
        return best_loss

class Default(TemplateStrategy):
    def __init__(self, m, n, seed=None):
        super(Default, self).__init__(seed)

        self._params = self.prng.rand(m*n)
        self.shape = (m, n)

    def strategy(self):
        A = self._params.reshape(self.shape)
        return matrix.EkteloMatrix(A)

    def _set_workload(self, W):
        self._WtW = W.gram().dense_matrix()
    
    def _loss_and_grad(self, params):
        WtW = self._WtW
        A = params.reshape(self.shape)
        sums = np.sum(np.abs(A), axis=0)
        col = np.argmax(sums)
        F = sums[col]**2
        # note: F is not differentiable, but we can take subgradients
        dF = np.zeros_like(A)
        dF[:,col] = np.sign(A[:,col])*2*sums[col]
        AtA = A.T.dot(A)
        AtA1 = np.linalg.pinv(AtA)
        M = WtW.dot(AtA1)
        G = np.trace(M)
        dX = -AtA1.dot(M)
        dG = 2*A.dot(dX)
        dA = dF*G + F*dG
        return F*G, dA.flatten()

class PIdentity(TemplateStrategy):
    """
    A PIdentity strategy is a strategy of the form (I + B) D where D is a diagonal scaling matrix
    that depends on B and ensures uniform column norm.  B is a p x n matrix of free parameters.
    """
    def __init__(self, p, n, seed=None):
        """
        Initialize a PIdentity strategy
        :param p: the number of non-identity queries
        :param n: the domain size
        """
        super(PIdentity, self).__init__(seed)

        self._params = self.prng.rand(p*n)
        self.p = p
        self.n = n

    def strategy(self):
        B = sparse.csr_matrix(self._params.reshape(self.p, self.n))
        I = sparse.eye(self.n, format='csr')
        A = sparse.vstack([I, B], format='csr')
        return matrix.EkteloMatrix(A / A.sum(axis=0))

    def _AtA1(self):
        B = np.reshape(self._params, (self.p,self.n))
        scale = 1.0 + np.sum(B, axis=0)
        R = np.linalg.inv(np.eye(self.p) + B @ B.T) # O(k^3)
        return (np.eye(self.n) - B.T @ R @ B)*scale*scale[:,None]
 
    def _set_workload(self, W):
        self._WtW = W.gram().dense_matrix()
        
    def _loss_and_grad(self, params):
        WtW = self._WtW
        p, n = self.p, self.n

        B = np.reshape(params, (p,n))
        scale = 1.0 + np.sum(B, axis=0)
        try: R = np.linalg.inv(np.eye(p) + B.dot(B.T)) # O(k^3)
        except: return np.inf, np.zeros_like(params)
        C = WtW * scale * scale[:,None] # O(n^2)
        M1 = R.dot(B) # O(n k^2)
        M2 = M1.dot(C) # O(n^2 k)
        M3 = B.T.dot(M2) # O(n^2 k)
        M4 = B.T.dot(M2.dot(M1.T)).dot(B) # O(n^2 k)

        Z = -(C - M3 - M3.T + M4) * scale * scale[:,None] # O(n^2)

        Y1 = 2*np.diag(Z) / scale # O(n)
        Y2 = 2*(B/scale).dot(Z) # O(n^2 k)
        g = Y1 + (B*Y2).sum(axis=0) # O(n k)

        loss = np.trace(C) - np.trace(M3)
        grad = (Y2*scale - g) / scale**2
        if loss < 0:
            return np.inf, np.zeros_like(params)
        return loss, grad.flatten()

class AugmentedIdentity(TemplateStrategy):
    """
    An AugmentedIdentity strategy is like a PIdentity strategy with additional structure imposed.
    The template is defiend by a p x n matrix of non-negative integers P.  Each unique nonzero entry
    of this matrix P refers to a free parameter that can be optimized.  An entry that is 0 in P is
    a structural zero in the strategy.  
    Example 1:
    A PIdentity strategy can be represented as an AugmentedIdentity strategy with 
    P = np.arange(1, p*n+1).reshape(p, n)
    
    Example 2:
    A strategy of the form w*T + I can be represented as an AugmentedIdentity strategy with
    P = np.ones((1, n), dtype=int)
    """
    def __init__(self, imatrix, seed=None):
        super(AugmentedIdentity, self).__init__(seed)

        self._imatrix = imatrix
        p, n = imatrix.shape
        num = imatrix.max()
        self._params = self.prng.rand(num)
        self._pid = PIdentity(p, n)

    def _set_workload(self, W):
        self._pid._set_workload(W)

    def strategy(self):
        params = np.append(0, self._params)
        B = params[self._imatrix]
        self._pid._params = B.flatten()
        return self._pid.strategy()

    def _AtA1(self):
        params = np.append(0, self._params)
        B = params[self._imatrix]
        self._pid._params = B.flatten()
        return self._pid._AtA1()

    def _loss_and_grad(self, params):
        params = np.append(0, params)
        B = params[self._imatrix]
        obj, grad = self._pid._loss_and_grad(B.flatten())
        grad2 = np.bincount(self._imatrix.flatten(), grad)[1:]
        return obj, grad2

class Static(TemplateStrategy):
    def __init__(self, strategy, approx = False, seed=None):
        super(Static, self).__init__(seed)

        self._strategy = strategy
        self._approx = approx

    def strategy(self):
        return self._strategy

    def optimize(self, W):
        A = self._strategy
        AtA = A.gram()
        AtA1 = AtA.pinv()
        WtW = W.gram()
        X = WtW @ AtA1

        if np.linalg.norm(WtW.dense_matrix() - (X @ AtA).dense_matrix()) >= 1e-5:
            return np.inf 

        if self._approx:
            delta = AtA.diag().max()
        else:
            delta = A.sensitivity()**2
        trace = X.trace()
        return delta * trace

class Kronecker(TemplateStrategy):
    def __init__(self, templates, seed=None):
        super(Kronecker, self).__init__(seed)

        self._templates = templates

    def strategy(self):
        return matrix.Kronecker([T.strategy() for T in self._templates])
        
    def optimize(self, W):
        if isinstance(W, matrix.Kronecker):
            loss = 1.0
            for subA, subW in zip(self._templates, W.matrices):
                loss *= subA.optimize(subW)
            return loss

        WtW = workload.sum_kron_canonical(W.gram())

        workloads = [[Q.dense_matrix() for Q in K.base.matrices] for K in WtW.matrices]
        weights = [K.weight for K in WtW.matrices]
        
        k = len(workloads)
        d = len(workloads[0])
        C = np.ones((d,k))

        for i in range(d):
            for j in range(k):
                C[i,j] = np.trace(workloads[j][i])
        for _ in range(10):
            #err = C.prod(axis=0).sum()
            for i in range(d):
                temp = self._templates[i]
                cs = weights * C.prod(axis=0) / C[i]
                What = sum(c*WtWs[i] for c, WtWs in zip(cs, workloads))
                What = workload.ExplicitGram(What / What.mean())
                temp.optimize(What)
                AtA1 = np.array(temp._AtA1())
                for j in range(k):
                    C[i,j] = np.sum(workloads[j][i] * AtA1)

        loss = (weights * C.prod(axis=0)).sum()
        return loss

class Union(TemplateStrategy):
    def __init__(self, templates, approx = False, seed=None):
        # expects workload to be a list of same length as templates
        # workload may contain subworkloads defined over different marginals of the data vector
        super(Union, self).__init__(seed)

        self._templates = templates
        self._weights = np.ones(len(templates)) / len(templates)
        self._approx = approx
    
    def strategy(self):
        # assumes each template returns a sensitivity 1 strategy
        return matrix.VStack([w * T.strategy() for w, T in zip(self._weights, self._templates)])
        
    def optimize(self, W):
        if isinstance(W, matrix.Weighted):
            W = W.base
        if isinstance(W, matrix.VStack):
            W = W.matrices
        assert isinstance(W, list), 'workload must be a list'
        assert len(W) == len(self._templates), 'length of workload list must match templates'
       
        errors = [] 
        for Ti, Wi in zip(self._templates, W):
            loss = Ti.optimize(Wi)
            errors.append(loss)
            #errors.append(error.expected_error(Wi, Ti.strategy()))

        if self._approx:
            weights = np.array(errors)**(1.0/4.0)
            weights /= np.linalg.norm(weights)
        else:
            weights = (2 * np.array(errors))**(1.0/3.0)
            weights /= weights.sum()

        self._weights = weights 

        return np.sum(errors / weights**2)

class Marginals(TemplateStrategy):
    def __init__(self, domain, approx = False, seed=None):
        super(Marginals, self).__init__(seed)

        self._domain = domain
        d = len(domain)
        self._params = self.prng.rand(2**len(domain))
        self._approx = approx
 
        self.gram = workload.MarginalsGram(domain, self._params**2)

    def strategy(self):
        if self._approx:
            weights = np.sqrt(self._params)
            weights /= np.linalg.norm(weights)
        else:
            weights = self._params / self._params.sum()
        return workload.Marginals(self._domain, weights)

    def _set_workload(self, W):
        WtW = workload.MarginalsGram.approximate(W.gram())
        self.weights = WtW.weights

    def _loss_and_grad(self, theta):
        n, d = np.prod(self._domain), len(self._domain)
        A = np.arange(2**d)
        mult = self.gram._mult
        weights = self.weights
        ones = np.ones(2**d)
       
        # TODO: accomodate (eps, delta)-DP
        if self._approx:
            delta = np.sum(theta)
            ddelta = 1
            theta2 = theta
        else: 
            delta = np.sum(theta)**2
            ddelta = 2*np.sum(theta)
            theta2 = theta**2
            
        X, XT = self.gram._Xmatrix(theta2)
        # D makes solve_triangular work for underdetermined systems
        D = sparse.diags(X.dot(ones)==0, dtype=float)
        phi = spsolve_triangular(X+D, weights, lower=False)
        if not np.allclose(X.dot(phi), weights):
            return np.inf, np.zeros_like(theta)
        ans = np.dot(phi, ones)*n
        
        dXvect = -spsolve_triangular(XT+D, ones*n, lower=True)
        # dX = outer(dXvect, phi)
        dtheta2 = np.array([np.dot(dXvect[A&b]*phi, mult[A|b]) for b in range(2**d)])
        if self._approx:
            dtheta = dtheta2
        else:
            dtheta = 2*theta*dtheta2
        return delta*ans, delta*dtheta + ddelta*ans

class McKennaConvex(TemplateStrategy):
    def __init__(self, n):
        self.n = n
        self._mask = np.tri(n, dtype=bool, k=-1)
        self._params = np.zeros(n*(n-1)//2)
        self.X = np.zeros((n,n))

    def strategy(self):
        tri = np.zeros((self.n,self.n))
        tri[self._mask] = self._params
        X = np.eye(self.n) + tri + tri.T
        A = np.linalg.cholesky(X).T
        return matrix.EkteloMatrix(A)

    def _set_workload(self, W):
        self.V = W.gram().dense_matrix().astype(float)
        self.W = W

    def _loss_and_grad(self, params):
        V = self.V
        X = self.X
        X.fill(0)
        #X = np.zeros((self.n,self.n))
        X[self._mask] = params
        X += X.T
        np.fill_diagonal(X, 1)

        zz, info0 = dpotrf(X, False, False)
        iX, info1 = dpotri(zz)
        iX = np.triu(iX) + np.triu(iX, k=1).T      
        if info0 != 0 or info1 != 0:
            #print('checkpt')
            return self._loss*100, np.zeros_like(params)
      
        loss = np.sum(iX * V) 
        G = -iX @ V @ iX
        g = G[self._mask] + G.T[self._mask]

        self._loss = loss
        #print(np.sqrt(loss / self.W.shape[0]))
        return loss, g#G.flatten() 

    def optimize(self, W):
        self._set_workload(W)

        eig, P = np.linalg.eigh(self.V)
        eig = np.real(eig)
        eig[eig < 1e-10] = 0.0
        X = P @ np.diag(np.sqrt(eig)) @ P.T
        X /= np.diag(X).max()
        x = X[self._mask]

        #x = np.eye(self.n).flatten()
        #bnds = [(1,1) if x[i] == 1 else (None, None) for i in range(x.size)]
        #x = self._params
      
        opts = { 'maxcor' : 1 } 
        res = optimize.minimize(self._loss_and_grad, x, jac=True, method='L-BFGS-B', options=opts)
        self._params = res.x
        #print(res)
        return res.fun       
  
class YuanConvex(TemplateStrategy):

    def __init__(self, seed=None):
        super(YuanConvex, self).__init__(seed)

    def optimize(self, W):
        V = W.gram().dense_matrix()

        accuracy = 1e-10
        max_iter_ls = 50
        max_iter_cg = 5
        theta = 1e-3
        
        beta = 0.5
        sigma = 1e-4
        n = V.shape[0]
        I = np.eye(n)
        X = I
        max_iter = 100
        V = V + theta*np.mean(np.diag(V))*I
        
        iX = I
        G = -V
        fcurr = np.sum((V*iX)**2)
        history = []

        for iter in range(1, max_iter+1):
            if iter == 1:
                D = -G
                np.fill_diagonal(D,0)
                j = 0
            else:
                D = np.zeros((n,n))
                Hx = lambda S: -iX.dot(S).dot(G) - G.dot(S).dot(iX)
                np.fill_diagonal(D, 0)
                R = -G - Hx(D)
                np.fill_diagonal(R, 0)
                P = R;
                rsold = np.sum(R**2)
                if np.sqrt(rsold) < 1e-8:
                    continue
                for j in range(1, max_iter_cg+1):
                    Hp = Hx(P)
                    alpha = rsold / np.sum(P * Hp)
                    D += alpha*P
                    np.fill_diagonal(D, 0)
                    R -= alpha*Hp
                    np.fill_diagonal(R, 0)
                    rsnew = np.sum(R**2)
                    if np.sqrt(rsnew) < 1e-8:
                        break
                    P = R + rsnew / rsold * P
                    rsold = rsnew

            delta = np.sum(D * G)
            X_old = X
            flast = fcurr
            history.append(fcurr)
            
            for i in range(1, max_iter_ls+1):
                alpha = beta**(i-1)
                X = X_old + alpha*D
                try:
                    iX = np.linalg.inv(X)
                    A = np.linalg.cholesky(X)
                except:
                    continue
                G = -iX.dot(V).dot(iX)
                fcurr = np.sum(V * iX)
                if fcurr <= flast + alpha*sigma*delta:
                    break

            #print(fcurr)

            if i==max_iter_ls:
                X = X_old
                fcurr = flast
                break
            if np.abs((flast - fcurr) / flast) < accuracy:
                break

        self.ans = np.linalg.cholesky(X).T
        return fcurr

    def strategy(self):
        return matrix.EkteloMatrix(self.ans)

def OPT0(n, approx=False):
    # note: temp1 and temp2 gives correct expected error for both approx=False and approx=True
    temp1 = Identity(n)
    temp2 = Total(n)
    if approx:
        temp3 = McKennaConvex(n)
        return BestTemplate([temp1, temp2, temp3]) 
    else:
        p = max(n//16, 1)
        temp3 = PIdentity(p, n)
        temp4 = IdTotal(n)
        return BestTemplate([temp1, temp2, temp3, temp4])

def DefaultKron(ns, approx=False):
    return Kronecker([OPT0(n, approx) for n in ns])

def DefaultUnionKron(ns, k, approx=False):
    return Union([DefaultKron(ns, approx) for _ in range(k)], approx)

def BestHD(ns, k, approx=False):
    temp1 = DefaultKron(ns, approx)
    temp2 = DefaultUnionKron(ns, k, approx)
    temp3 = Marginals(ns, approx)
    return BestTemplate([temp1, temp2, temp3])
 
def KronYuan(ns):
    return Kronecker([YuanConvex() for _ in ns])

def KronPIdentity(ps, ns):
    """
    Builds a template strategy of the form A1 x ... x Ad where each Ai is a PIdentity template
    :param ps: the number of p queries in each dimension
    :param ns: the domain size of each dimension
    """
    OPT0 = lambda p, n: BestTemplate([Identity(n), Total(n), PIdentity(p, n)])
    return Kronecker([OPT0(p, n) for p,n in zip(ps, ns)])

def UnionKron(ps, ns):
    """
    Builds a template strategy that is a union of Kronecker products, where each
    kron product is a PIdentity strategy

    :param ps: a table of p values of size k x d where k is number of strategies in union and d in number of dimensions
    :param ns: the domain size of each dimension (length d tuple)
    """
    return Union([KronPIdentity(p, ns) for p in ps])

def RangeTemplate(n, start=32, branch=4, shared=False):
    """
    Builds a template strategy for range queries with queries that have structural zeros 
    everywhere except at indices at [i, i+w) where w is the width of the query and ranges from
    start to n in powers of branch and i is a multiple of w/2.
    :param n: the domain size
    :param start: the width of the smallest query
    :param branch: the width multiplying factor for larger queries
    :param shared: flag to determine if parameters should be shared for queries of the same width
    Example:
    RangeTemplate(16, start=8, branch=2) builds a strategy template with four augmented queries that have structural zeros everywhere except in the intervals indicated below:
    1. [0,8)
    2. [4,12)
    3. [8,16)
    4. [0,16)
    """
    rows = []
    width = start
    idx = 1
    while width <= n:
        for i in range(0, n-width//2, width//2):
            row = np.zeros(n, dtype=int)
            row[i:i+width] = np.arange(width) + idx
            if not shared: idx += width
            rows.append(row)
        if shared: idx += width
        width *= branch
    return AugmentedIdentity(np.vstack(rows))

def IdTotal(n):
    """ Build a single-parameter template strategy of the form w*Total + Identity """
    P = np.ones((1,n), dtype=int)
    return AugmentedIdentity(P)

def Identity(n):
    """ Builds a template strategy that is always Identity """
    return Static(workload.Identity(n))

def Total(n):
    """ Builds a template strategy that is always Total """
    return Static(workload.Total(n))
