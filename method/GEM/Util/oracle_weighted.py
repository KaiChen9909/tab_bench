import numpy as np
from gurobipy import *


def solve(queries, queries_weights, neg_queries, neg_queries_weights, sigma, domain, alpha):
    """
    Gurobi solver for k-way KWayMarginals
    """
    dim = sum([p for p in domain.shape])
    reg_num_queries = queries.shape[0] if queries is not None else 0
    neg_num_queries = neg_queries.shape[0] if neg_queries is not None else 0
    num_queries = reg_num_queries + neg_num_queries
    # assert sigma.shape[0] == dim

    c = {}
    x = {}
    model = Model("BestResponse")

    model.setParam("OutputFlag", 0)
    model.setParam("MIPGapAbs", alpha)
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
    obj1 = quicksum(c[i]*queries_weights[i] for i in range(reg_num_queries))
    obj2 = quicksum(c[i+reg_num_queries]*neg_queries_weights[i] for i in range(neg_num_queries))
    obj3 = quicksum(x[i] * sigma[i] for i in range(dim))
    #print("sigma ", sigma)
    model.setObjective(obj1+obj2-obj3, GRB.MAXIMIZE)
    """
    Each features must have 1
    """
    cur = 0
    for f, sz in enumerate(domain.shape):
        model.addConstr(quicksum(x[j] for j in range(cur, cur + sz)) == 1)
        cur += sz
    """
    if x[a]  & x[b] & x[c] then c[i] <-- 1
    """
    for i in range(reg_num_queries):
        K = np.sum(queries[i,:])
        model.addConstr(quicksum(x[j]*queries[i,j] for j in range(dim)) >= K*c[i] - 1e-6)

    """
    if !x[a] | !x[b] | !x[c] then c[i] <-- 0
    """
    for i in range(neg_num_queries):
        model.addConstr(quicksum((1-x[j])*neg_queries[i, j] for j in range(dim)) >= c[reg_num_queries + i] - 1e-6)

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
