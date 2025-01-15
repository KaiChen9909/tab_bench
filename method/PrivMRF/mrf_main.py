import sys
target_path="./"
sys.path.append(target_path)

import method.PrivMRF.PrivMRF.utils.tools as tools
from method.PrivMRF.PrivMRF.domain import Domain
from method.PrivMRF.PrivMRF.main import run
import numpy as np

def prepare_domain(domain): 
    domain_new = {}
    i=0
    for k,v in domain.items():
        domain_new[i] = {"type": "discrete", "domain": v}
        i+=1
    return domain_new


def mrf_main(args, df, domain, rho, **kwargs):
    # should provide int data
    data = df.to_numpy()
    
    # domain of each attribute should be [0, 1, ..., max_value-1]
    # attribute name should be 0, ..., column_num-1.
    # json_domain = tools.read_json_domain('./preprocess/nltcs.json')
    json_domain = prepare_domain(domain)
    print(json_domain)
    domain = Domain(json_domain, list(range(data.shape[1])))

    # you may set hyperparameters or specify other settings here
    config = {}

    # train a PrivMRF, delta=1e-5
    # for other dp parameter delta, calculate the privacy budget 
    # with cal_privacy_budget() of ./PrivMRF/utils/tools.py 
    # and hard code the budget in privacy_budget() of ./PrivMRF/utils/tools.py 
    model = run(data, domain, attr_hierarchy=None, \
        exp_name='exp', rho=rho, p_config=config) 
    
    return {"mrf_generator": model}

    

