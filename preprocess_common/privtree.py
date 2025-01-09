################################################################

# This is the main implementation of PrivTree discretizer
# mostly from https://github.com/ruankie/differentially-private-range-queries

################################################################


import numpy as np
import pandas as pd

def laplace_noise(Lambda):
    return np.random.laplace(loc=0, scale=Lambda)

def get_domain_subdomains(domain):
    
    # domain = [left, right]
    # left = domain[0]
    # right = domain[1]
    
    # -----------
    # | q1 | q2 |
    # -----------

    dom_w = domain[1] - domain[0]
    
    q1 = [domain[0], domain[1] - dom_w/2]
    q2 = [domain[0] + dom_w/2, domain[1]]
    
    return q1, q2


def is_in_domain(x, left, right): #includes left and bottom border
        if (x >= left) and (x < right):
            return True
        else:
            return False

def count_in_domain(xs, domain):
    count = 0
    
    for i in range(xs.shape[0]):
        if is_in_domain(xs[i], domain[0], domain[1]):
            count += 1

    return count

def is_domain_partially_in_domain(v1, v0): 
    # checks if domain v1 is partially inside domain v0
    # domain v1 corners (TL = top left, TR = top right, BL = bottom left, BR = botom right)
    is_partially_in = False
    
    v0_L, v0_R = v0[0], v0[1]
    v1_L, v1_R = v1[0], v1[1]
    
    if is_in_domain(v1_L, v1_R, v0_L, v0_R):
        is_partially_in = True
    return is_partially_in


class privtree():
    # simple tree parameters
    # eps : dp param
    # theta : 50 #min count per domain
    # notice that the data inputed into this mechanism all need preprocess, which means K = data.shape[1]

    def __init__(self, rho, theta=0, domain_margin=1e-2, seed=0):
        self.rho = rho
        self.theta = theta 
        self.domain_margin = domain_margin
        self.seed = seed

    def fit(self, data):
        self.split = []

        if data.ndim > 1:
            K = data.shape[1]
            self.lam, self.delta = self.calculate_param(self.rho, K)
            for i in range(data.shape[1]):
                x = data[:, i]
                self.split.append(self.tree_main(x, self.lam*K, self.delta*K))
        else:
            K = 1
            self.lam, self.delta = self.calculate_param(self.rho, K)
            self.split = self.tree_main(data, self.lam, self.delta)
    

    def transform(self, data):
        if data.ndim > 1:
            transformed_data = np.empty_like(data, dtype=int)
            for i in range(data.shape[1]):
                x = data[:, i]
                transformed_data[:, i] = self.transform_data(x, self.split[i])
        else:
            transformed_data = self.transform_data(data, self.split)
        
        return transformed_data


    def fit_transform(self, data):
        self.split = []

        if data.ndim > 1:
            transformed_data = np.empty_like(data, dtype=int)
            K = data.shape[1]
            self.lam, self.delta = self.calculate_param(self.rho, K)
            for i in range(data.shape[1]):
                x = data[:, i]
                self.split.append(self.tree_main(x, self.lam*K, self.delta*K))
                transformed_data[:, i] = self.transform_data(x, self.split[i])
        else:
            K = 1
            self.lam, self.delta = self.calculate_param(self.rho, K)
            self.split = self.tree_main(data, self.lam, self.delta)
            transformed_data = self.transform_data(data, self.split)

        return transformed_data


    def inverse_transform(self, data):
        inversed_data = np.empty_like(data, dtype=float)

        if data.ndim > 1:
            for i in range(data.shape[1]):
                inversed_data[:, i] = self.inverse_data(data[:,i], self.split[i])
        else:
            inversed_data = self.inverse_data(data, self.split)
            
        return inversed_data


    def tree_main(self, x, lam, delta):
        domains = []
        unvisited_domains = []
        counts = []
        noisy_counts = []
        tree_depth = 0

        x_min, x_max = np.min(x), np.max(x)
        v0 = [x_min, x_max + self.domain_margin] 
        unvisited_domains.append(v0)

        # create subdomains where necessary
        while not not unvisited_domains: # while unvisited_domains is not empty
            for unvisited in unvisited_domains:
                # calculate count and noisy count
                count = count_in_domain(x, unvisited)
                b = count - (delta*tree_depth)
                b = max(b, (self.theta - delta))
                noisy_b = b + laplace_noise(lam)

                if (noisy_b > self.theta): #split if condition is met
                    v1, v2 = get_domain_subdomains(unvisited)
                    unvisited_domains.append(v1)
                    unvisited_domains.append(v2)
                    unvisited_domains.remove(unvisited)
                    tree_depth += 1
                else:
                    unvisited_domains.remove(unvisited)
                    counts.append(count)
                    noisy_counts.append(noisy_b)
                    domains.append(unvisited)

        return sorted(domains, key=lambda x: x[0])


    def transform_data(self, data_col, domains):
        conditions = [(data_col >= a) & (data_col < b) for a, b in domains]
        choices = list(range(len(domains)))

        return np.select(conditions, choices, default=-1)
    
    def inverse_data(self, data_col, domains):
        return np.array([np.random.uniform(domains[int(i)][0], domains[int(i)][1]) for i in data_col])


    def calculate_param(self, rho, K):
        beta = 2 
        lam = (2*beta - 1)/(beta - 1) * np.sqrt(K/(2*rho))
        delta = lam * np.log(beta)

        return lam, delta
        