import numpy as np
from scipy import optimize

"""A class representing datasets
It stores the num_sample transition data of the 
birth-death-migration process parameters lamda,mu and m over a a time period of length cutoff_time
where the initial_states are stored in initial state (an array of size num_sample)

You can use ds.dump(filename) to store dataset in a file

to load use ds = load(filename)
"""


class Dataset:
    def __init__(self, lamda, mu, m, cutoff_time, num_sample, initial_state, transition_data=None):
        self.lamda = lamda
        self.mu = mu
        self.m = m
        self.num_sample = num_sample
        self.cutoff_time = cutoff_time
        self.initial_state = initial_state
        self.transition_data = transition_data

    def __repr__(self):
        return "{\n\t" + "lamda: {},\
        \n\tmu: {},\
        \n\tm: {},\
        \n\tinitial_state: {},\
        \n\tnum_sample: {},\
        \n\tcutoff_time: {},\
        \n\ttransition_data: {}\n".format(self.lamda, self.mu, self.m, self.initial_state,
                                          self.cutoff_time, self.num_sample,
                                          self.transition_data) + "}"


def get_dataset(tree):
    td = np.zeros(shape=(tree.number_of_species * tree.number_of_sites, 2), dtype=np.int32)
    td[:, 0] = tree.kmer_count[1:tree.number_of_species + 1].flatten()
    ds = Dataset(tree.lamda, tree.mu, tree.m, 0, td.shape[0], None, transition_data=td)
    return ds


def create_cnt_array(data):
    data = data.transition_data[:, 0]
    arr = np.zeros(shape=(np.max(data) + 1,), dtype=np.int32)
    for x in data:
        arr[x] += 1
    return arr


def gen_ln_pi_array(x, y, max_kmer_cnt):
    pi = np.zeros(shape=(max_kmer_cnt + 1,))
    pi[0] = y * np.log(1 - x)

    for i in range(1, max_kmer_cnt + 1):
        pi[i] = pi[i - 1] + np.log((i - 1) / i * x + 1 / i * x * y)
    return pi


def gen_gradient_ln_pi_array(x, y, max_kmer_cnt):
    gpi = np.zeros(shape=(2, max_kmer_cnt + 1))
    gpi[0, 0] = -y / (1 - x)
    gpi[1, 0] = np.log(1 - x)

    for i in range(1, max_kmer_cnt + 1):
        gpi[:, i] = gpi[:, i - 1]
        gpi[0, i] += 1 / x
        gpi[1, i] += 1 / ((i - 1) + y)

    return gpi


def gen_hessian_ln_pi_array(x, y, max_kmer_count):
    ans = np.ndarray(shape=(2, 2, max_kmer_count + 1))
    ans[0, 0, 0] = -y / (1 - x) ** 2
    ans[0, 1, 0] = -1 / (1 - x)
    ans[1, 0, 0] = ans[0, 1, 0]
    ans[1, 1, 0] = 0

    for i in range(1, max_kmer_count + 1):
        c1 = (i - 1) / i
        c2 = 1 / i
        ans[:, :, i] = ans[:, :, i - 1]
        inc = np.ndarray(shape=(2, 2))
        inc[0, 0] = 1 / x ** 2
        inc[0, 1] = 0
        inc[1, 0] = 0
        inc[1, 1] = c2 ** 2 / (c1 + c2 * y) ** 2
        ans[:, :, i] -= inc
    return ans


def modified_log_likelihood(x, y, cnt_array):
    max_kmer_cnt = len(cnt_array) - 1
    pi = gen_ln_pi_array(x, y, max_kmer_cnt)
    return np.dot(pi, cnt_array)


def gradient_log_likelihood(x, y, cnt_array):
    max_kmer_cnt = len(cnt_array) - 1
    gpi = gen_gradient_ln_pi_array(x, y, max_kmer_cnt)
    return np.matmul(gpi, cnt_array)


def hessian_log_likelihood(x, y, cnt_array):
    max_kmer_cnt = len(cnt_array) - 1
    gpi = gen_hessian_ln_pi_array(x, y, max_kmer_cnt)
    ans = np.zeros(shape=(2, 2))
    for i in range(len(cnt_array)):
        ans += gpi[:, :, i] * cnt_array[i]
    return ans


def initializer():
    return np.array([np.random.uniform(), np.random.uniform(0, 2)])


class BDPSParameterEstimator:
    def __init__(self, ds):
        self.cnt_arr = create_cnt_array(ds)

    def likelihood_func(self, theta):
        return -float(modified_log_likelihood(theta[0], theta[1], self.cnt_arr))

    def gradient(self, theta):
        return -gradient_log_likelihood(theta[0], theta[1], self.cnt_arr).astype(float)

    def hessian(self, theta):
        return -hessian_log_likelihood(theta[0], theta[1], self.cnt_arr).astype(float)

    def estimate_parameter(self, initializer, eps=0.0000001, number_of_trials=4):
        ans = initializer()
        vmin = self.likelihood_func(ans)
        bounds = optimize.Bounds([eps, eps], [1 - eps, np.inf])
        for i in range(number_of_trials):
            gg = optimize.minimize(self.likelihood_func, initializer(), method='trust-constr',
                                         jac=self.gradient, hess=self.hessian, options={'verbose': 1}, bounds=bounds)
            g = gg['x']
            v = self.likelihood_func(g)
            if v < vmin:
                ans = g
                vmin = v
        ans[1] = ans[0] * ans[1]
        return ans
