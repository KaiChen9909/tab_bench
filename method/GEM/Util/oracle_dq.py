import numpy as np
from gurobipy import *
import tempfile

def dualquery_best_response(queries, neg_queries, dim, domain, alpha):
    """
    Gurobi solver for k-way KWayMarginals
    """
    # assert len(queries.shape) == 2, "{}".format(queries)
    # assert len(neg_queries.shape) == 2, "{}".format(neg_queries)
    reg_num_queries = queries.shape[0]
    neg_num_queries = neg_queries.shape[0]
    num_queries = reg_num_queries + neg_num_queries
    if  num_queries== 0:
        return np.zeros(dim)

    c = {}
    x = {}
    model = Model("DQBestResponse")

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
    obj1 = quicksum(c[i] for i in range(num_queries))
    model.setObjective(obj1 , GRB.MAXIMIZE)
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
    if !x[a] | !x[b] | x[c] then c[i] <-- 0
    """
    for i in range(neg_num_queries):
        model.addConstr(quicksum((1-x[j])*neg_queries[i,j] for j in range(dim)) >= c[i+reg_num_queries]- 1e-6)

    model.optimize()

    x_sync = [int(x[i].X + 0.5) for i in range(dim)]
    for x in x_sync:
        assert x ==0 or x == 1
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
    mipGapAbs = np.abs(model.getAttr(GRB.Attr.ObjVal)-model.getAttr(GRB.Attr.ObjBound))
    # print("MIPGap = {}".format(model.getAttr(GRB.Attr.MIPGap)))
    # print("MIPGap = {}".format(model.getAttr(GRB.Attr.MIPGap)))
    return x_sync
