import math
import random
import numpy
import copy

# Auxiliary functions
def sgn(x):
    if x > 0:
        return 1
    if x == 0:
        return 0
    return -1

def dabs(x):
    return -x if x < 0 else x

# Tree heap operations
class Node:
    def __init__(self, key):
        self.count = 1
        self.key = key
        self.fix = random.randint(0, 2**31 - 1)  # Random number to serve as a fix
        self.sum = key
        self.left = None
        self.right = None

class Res:
    def __init__(self):
        self.lct = 0
        self.rct = 0
        self.lsum = 0
        self.rsum = 0

def left_rot(x):
    y = x.right
    if y is None:
        return
    x.right = y.left
    y.left = x
    x = y

    x.count = 1
    x.sum = x.key
    if x.left is not None:
        x.count += x.left.count
        x.sum += x.left.sum
    if x.right is not None:
        x.count += x.right.count
        x.sum += x.right.sum

def right_rot(x):
    y = x.left
    if y is None:
        return
    x.left = y.right
    y.right = x
    x = y

    x.count = 1
    x.sum = x.key
    if x.left is not None:
        x.count += x.left.count
        x.sum += x.left.sum
    if x.right is not None:
        x.count += x.right.count
        x.sum += x.right.sum

def insert(x, k):
    if x is None:
        return Node(k)
    elif k < x.key:
        x.count += 1
        x.sum += k
        x.left = insert(x.left, k)
        if x.left.fix < x.fix:
            right_rot(x)
    else:
        x.count += 1
        x.sum += k
        x.right = insert(x.right, k)
        if x.right.fix < x.fix:
            left_rot(x)
    return x

def delnode(x):
    if x is None:
        return
    delnode(x.left)
    delnode(x.right)

def remove(x, k):
    if x is None:
        return False
    if k < x.key:
        if remove(x.left, k):
            x.count -= 1
            x.sum -= k
    elif k > x.key:
        if remove(x.right, k):
            x.count -= 1
            x.sum -= k
    else:
        if x.left is None or x.right is None:
            y = x
            x = x.left if x.left else x.right
            del y
        elif x.left.fix < x.right.fix:
            right_rot(x)
            if remove(x.right, k):
                x.count -= 1
                x.sum -= k
        else:
            left_rot(x)
            if remove(x.left, k):
                x.count -= 1
                x.sum -= k
    return True

def search(x, e):
    if x is None:
        return Res()
    if e < x.key:
        cur = search(x.left, e)
        cur.rct += 1
        cur.rsum += x.key
        if x.right is not None:
            cur.rct += x.right.count
            cur.rsum += x.right.sum
    else:
        cur = search(x.right, e)
        cur.lct += 1
        cur.lsum += x.key
        if x.left is not None:
            cur.lct += x.left.count
            cur.lsum += x.left.sum
    return cur

# Noisy L1 partition with interval buckets of size 2^k
def L1partition_approx(x, n, epsilon, ratio, seed):
    score = [[0] * (int(math.log(n, 2)) + 1) for _ in range(n)]
    invepsilon1 = 1.0 / (epsilon * ratio)
    invepsilon2 = 1.0 / (epsilon - epsilon * ratio)
    invn = 1.0 / n

    random.seed(seed)

    len_ = 1
    off = 0
    while len_ <= n:
        root = Node(x[0])
        total_sum = x[0]
        invlen = 1.0 / len_
        for i in range(1, len_):
            insert(root, x[i])
            total_sum += x[i]

        for i in range(len_ - 1, n):
            avg = total_sum / len_
            r = 0.5 - random.random()
            lap = 0
            if len_ > 1:
                lap = (2.0 - invlen - invn) * invepsilon1 * sgn(r) * math.log(1.0 - 2.0 * dabs(r))
            loc = search(root, avg)
            score[i][off] = (loc.lct - loc.rct) * avg - loc.lsum + loc.rsum
            score[i][off] += invepsilon2 + lap
            score[i][off] = max(0, score[i][off])
            del loc
            if i < n - 1:
                total_sum += x[i + 1] - x[i - len_ + 1]
                insert(root, x[i + 1])
                remove(root, x[i - len_ + 1])
        len_ <<= 1
        off += 1
        delnode(root)

    cumscore = [0] * (n + 1)
    lbound = [-1] * (n + 1)
    cumscore[0] = 0
    lbound[0] = -1
    for i in range(n):
        cumscore[i + 1] = cumscore[i] + score[i][0]
        lbound[i + 1] = i
        for j in range(1, int(math.log(i + 1, 2)) + 1):
            len_ = 2 ** j
            if len_ <= i + 1:
                curscore = cumscore[i - len_ + 1] + score[i][j]
                if curscore <= cumscore[i + 1]:
                    lbound[i + 1] = i - len_ + 1
                    cumscore[i + 1] = curscore

    j = n
    hist = [0] * (n+1)
    for i in range(n + 1):
        hist[i] = j
        j = lbound[j]
    return hist

# Noisy L1 partition with all interval buckets
def cumabserr(x, n):
    err = [0] * n
    total_sum = x[0]
    root = Node(x[0])
    err[0] = 0
    for i in range(1, n):
        total_sum += x[i]
        avg = total_sum / (i + 1)
        insert(root, x[i])
        loc = search(root, avg)
        err[i] = (loc.lct - loc.rct) * avg - loc.lsum + loc.rsum
        del loc
    delnode(root)
    return err

def L1partition(x, n, epsilon, ratio, seed):
    score = [cumabserr(x[i:], n - i) for i in range(n)]
    invepsilon1 = 1.0 / (epsilon * ratio)
    invepsilon2 = 1.0 / (epsilon - epsilon * ratio)
    
    random.seed(seed)

    for i in range(n):
        for j in range(n - i):
            r = 0.5 - random.random()
            lap = 0
            if j > 0:
                lap = (2.0 - 1.0 / (j + 1) - 1.0 / n) * invepsilon1 * sgn(r) * math.log(1.0 - 2.0 * dabs(r))
            score[i][j] += invepsilon2 + lap
            score[i][j] = max(0, score[i][j])

    cumscore = [0] * (n + 1)
    lbound = [-1] * (n + 1)
    cumscore[0] = 0
    lbound[0] = -1
    for i in range(n):
        cumscore[i + 1] = cumscore[i] + score[i][0]
        lbound[i + 1] = i
        for j in range(i):
            curscore = cumscore[j] + score[j][i - j]
            if curscore < cumscore[i + 1]:
                lbound[i + 1] = j
                cumscore[i + 1] = curscore

    j = n
    hist = [0] * (n+1)
    for i in range(n + 1):
        hist[i] = j
        j = lbound[j]
    return hist


registry = {}

def register(name):
	def wrap(cls):
		force_bound = False
		if '__init__' in cls.__dict__:
			cls.__init__.func_globals[name] = cls
			force_bound = True
		try:
			registry[name] = cls
		finally:
			if force_bound:
				del cls.__init__.func_globals[name]
		return cls
	return wrap


def L1partition_fn(x, epsilon, ratio=0.5, gethist=False):
	"""Compute the noisy L1 histogram using all interval buckets

	Args:
		x - list of numeric values. The input data vector
		epsilon - double. Total private budget
		ratio - double in (0, 1). use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
				the count in each partition
		gethist - boolean. If set to truth, return the partition directly (the privacy budget used is still ratio*epsilon)

	Return:
		if gethist == False, return an estimated data vector. Otherwise, return the partition
	"""
	n = len(x)
	hist = L1partition(x, n, epsilon, ratio, numpy.random.randint(500000))
	hatx = numpy.zeros(n)
	rb = n
	if gethist:
		bucks = []
		for lb in hist[1:]:
			bucks.insert(0, [lb, rb])
			rb = lb
			if lb == 0:
				break
		return bucks
	else:
		for lb in hist[1:]:
			hatx[lb:rb] = max(0, sum(x[lb:rb]) + numpy.random.laplace(0, 1.0/(epsilon*(1-ratio)), 1)) / float(rb - lb)
			rb = lb
			if lb == 0:
				break

		return hatx


def L1partition_approx_fn(x, epsilon, ratio=0.5, gethist=False):
	"""Compute the noisy L1 histogram using interval buckets of size 2^k

	Args:
		x - list of numeric values. The input data vector
		epsilon - double. Total private budget
		ratio - double in (0, 1) the use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
				the count in each partition
		gethist - boolean. If set to truth, return the partition directly (the privacy budget used is still ratio*epsilon)

	Return:
		if gethist == False, return an estimated data vector. Otherwise, return the partition
	"""
	n = len(x)
	hist = L1partition_approx(x, n, epsilon, ratio, numpy.random.randint(500000))
	hatx = numpy.zeros(n)
	rb = n
	if gethist:
		bucks = []
		for lb in hist[1:]:
			bucks.insert(0, [lb, rb-1])
			rb = lb
			if lb == 0:
				break
		return bucks
	else:
		for lb in hist[1:]:
			hatx[lb:rb] = max(0, sum(x[lb:rb]) + numpy.random.laplace(0, 1.0/(epsilon*(1-ratio)), 1)) / float(rb - lb)
			rb = lb
			if lb == 0:
				break

		return hatx



class partition_engine(object):
	"""The template class for partition engines."""

	@staticmethod
	def Run(x, epsilon, ratio):
		"""Run templated for partition engine.

		x - the input dataset
		epsilon - the total privacy budget
		ratio - the ratio of privacy budget used for partitioning.
		"""
		raise NotImplementedError('A Run method must be implemented'
								  ' for a partition engine.')


@register('l1partition')
class l1_partition(partition_engine):
	"""Use the L1 partition method."""

	@staticmethod
	def Run(x, epsilon, ratio):
		return L1partition_fn(x, epsilon, ratio, gethist=True)


@register('l1approx')
class l1_partition_approx(partition_engine):
	"""Use the approximate L1 partition method."""

	@staticmethod
	def Run(x, epsilon, ratio):
		return L1partition_approx_fn(x, epsilon, ratio, gethist=True) 
     

def interval_transform(data, interval):
    bin_edges = numpy.array([b for _, b in interval])
    encoded_array = numpy.searchsorted(bin_edges, data, side="right")

    return encoded_array

def interval_inverse_transform(data, interval):
    lower_bounds = interval[data, 0]
    upper_bounds = interval[data, 1]
    decoded_array = numpy.random.uniform(lower_bounds, upper_bounds) 

    return decoded_array

    
class dawa():
    def __init__(self, rho):
        self.rho = rho 

    def fit(self, data):
        epsilon = math.sqrt(2*self.rho/data.shape[1])
        self.box = []

        for i in range(data.shape[1]):
            intial_hist, initial_divide = numpy.histogram(data[:,1], bins=1000)
            initial_divide[-1] = initial_divide[-1] + 1e-2
            partition = l1_partition_approx.Run(intial_hist, 2*epsilon, 0.5)
            
            box_i = []
            for interval in partition:
                box_i.append([initial_divide[interval[0]], initial_divide[interval[1]+1]]) 
            self.box.append(numpy.array(box_i))

    def transform(self, data):
        encoded_data = numpy.empty_like(data)
        for i in range(data.shape[1]):
            encoded_data[:, i] = interval_transform(encoded_data[:, i], self.box[i])

        return encoded_data

    def fit_transform(self, data):
        epsilon = math.sqrt(2*self.rho/data.shape[1])
        self.box = []

        for i in range(data.shape[1]):
            intial_hist, initial_divide = numpy.histogram(data[:,1], bins=1000)
            initial_divide[-1] = initial_divide[-1] + 1e-2
            partition = l1_partition_approx.Run(intial_hist, 2*epsilon, 0.5)

            box_i = []
            for interval in partition:
                box_i.append([initial_divide[interval[0]], initial_divide[interval[1]+1]]) 
            self.box.append(numpy.array(box_i))

        encoded_data = numpy.empty_like(data)
        for i in range(data.shape[1]):
            encoded_data[:, i] = interval_transform(encoded_data[:, i], self.box[i])

        return encoded_data
    
    
    def inverse_transform(self, data):
        decoded_data = numpy.empty_like(data)
        for i in range(data.shape[1]):
            decoded_data[:, i] = interval_inverse_transform(decoded_data[:, i], self.box[i]) 
        
        return decoded_data
