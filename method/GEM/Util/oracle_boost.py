import numpy as np
from gurobipy import *


# def solve(queries, neg_queries, sigma, domain, alpha):
def solve(queries_w, weights, domain, randomize_weight=1):
    """
    """
    reg_num_queries, dim = queries_w.shape
    num_queries = reg_num_queries
    if  num_queries== 0:
        return np.zeros(dim)

    c = {}
    x = {}
    model = Model("BestResponse")

    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 60)

    for i in range(num_queries):
        """
        c[i]: Indicator for the ith query; ===> q_i(x) = 1
        """
        c[i] = model.addVar(vtype=GRB.BINARY, name="c_{}".format(i))
    for i in range(dim):
        """
        x[i]: Optimization variables
        """
        x[i] = model.addVar(vtype=GRB.BINARY, name="x_{}".format(i))
    model.update()

    ## Objective
    obj1 = quicksum(weights[i]*c[i] for i in range(num_queries))
    perturb = np.random.rand(dim)*randomize_weight
    obj2 = quicksum(x[i] * perturb[i] for i in range(dim))  # secondary objective
    #print("sigma ", sigma)
    model.setObjective(obj1+obj2, GRB.MAXIMIZE)
    """
    Each features must have 1
    """
    cur = 0
    for f, sz in enumerate(domain.shape):
        model.addConstr(quicksum(x[j] for j in range(cur, cur + sz)) == 1)
        cur += sz

    for i in range(num_queries):
        K = np.sum(queries_w[i,:])
        # if query_type[i] == 1:
        """
        if x[a]  & x[b] & x[c] then c[i] <-- 1
        """
        model.addConstr(quicksum(x[j]*queries_w[i,j] for j in range(dim)) >= K*c[i] - 1e-6)
        # else:
        #     """
        #     if !x[a] | !x[b] | !x[c] then c[i] <-- 1
        #     """
        #     model.addConstr(quicksum((1 - x[j]) * queries_w[i, j] for j in range(dim)) >= c[i] - 1e-6)

    model.optimize()

    x_sync = [int(x[i].X + 0.5) for i in range(dim)]
    for x in x_sync:
        assert x ==0 or x == 1
    assert np.sum(x_sync) == len(domain.shape) , "sum(x_sync) = {}, len(domain) = {}".format(np.sum(x_sync), len(domain.shape))
    assert len(x_sync) == dim , "len(x_sync) = {}, dim = {}".format(len(x_sync), dim)

    """
    Check Constraints
    """
    # for i, q in enumerate(queries):
    #     K = len(q.ind)
    #     satisfied = c[i].x >=  0.5
    #     sum = 0
    #     sum_neg = 0
    #     for (col, val) in zip(q.ind, q.val):
    #         sum += x_sync[col] if val == 1 else 1 - x_sync[col]
    #         sum_neg += 1-x_sync[col] if val == 1 else x_sync[col]
    #     if satisfied:
    #         if not q.negated:
    #             assert sum == K, "sum = {}".format(sum)
    #         else:
    #             assert sum_neg >= 1, "sum_neg = {}".format(sum_neg)
    #     else:
    #         if not q.negated:
    #             assert sum < K, "sum = {}".format(sum)
    #         else:
    #             assert sum_neg == 0, "sum = {}".format(sum_neg)

    """
    Synthetic record
    """
    return x_sync
