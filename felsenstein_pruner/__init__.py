import numpy as np
from scipy import linalg

import bdps_parameter_estimator as bpe


def gen_censored_linear_bdps_qmat(lamda, m, mu, max_kmer_cnt, last_back_edge=None):
    Q = np.zeros(shape=(max_kmer_cnt + 1, max_kmer_cnt + 1))
    Q[0][0] = -m
    Q[0][1] = m

    for i in range(1, max_kmer_cnt):
        Q[i][i - 1] = i * mu
        Q[i][i + 1] = i * lamda + m
        Q[i][i] = -(Q[i][i - 1] + Q[i][i + 1])

    if last_back_edge is None:
        last_back_edge = max_kmer_cnt * mu

    Q[max_kmer_cnt][max_kmer_cnt - 1] = last_back_edge
    Q[max_kmer_cnt][max_kmer_cnt] = -last_back_edge
    return Q


def gen_pi_array(q_matrix):
    max_kmer_count = q_matrix.shape[0] - 1
    qt = np.transpose(q_matrix)
    A = np.ones(shape=(max_kmer_count + 2, max_kmer_count + 1))
    A[:-1] = qt

    b = np.zeros(shape=(max_kmer_count + 2))
    b[-1] = 1

    return linalg.lstsq(A, b)[0]


class FelsensteinPrunner:
    def __init__(self, tree, qmat=None, pi=None):
        self.tree = tree
        self.Q = qmat
        self.max_kmer_cnt = qmat.shape[0]
        if pi is None:
            pi = gen_pi_array(qmat)
        self.pi = pi

        self.dp = np.zeros(shape=(tree.n, tree.number_of_sites, self.max_kmer_cnt))
        self.lg_factor = np.zeros(shape=(tree.n, tree.number_of_sites))
        self.parents = np.ndarray(shape=(tree.n,), dtype=np.int32)

    def transition(self, src, dest, time):
        return linalg.expm((self.Q * time))[src][dest]

    def get_transition_matrix(self, time):
        return linalg.expm(self.Q * time)

    def relax_node(self, tree, root, parent):
        self.dp[root][:][:] = 1
        self.lg_factor[root][:] = 0
        for e in tree.adjList[root]:
            if e is None or e.id == -1:
                break
            if e.dest == parent:
                continue

            v = e.dest
            w = e.length
            self.lg_factor[root][:] += self.lg_factor[v][:]

            transition_matrix = np.transpose(self.get_transition_matrix(w))
            self.dp[root] *= np.matmul(self.dp[v], transition_matrix)

        s = np.sum(self.dp[root], axis=1)
        self.lg_factor[root] += -np.log(s)

        for i in range(tree.number_of_sites):
            self.dp[root][i] = self.dp[root][i] / s[i]

    def compute_likelihood_helper(self, tree, root, parent):
        self.parents[root] = parent
        if tree.degree[root] == 0 or tree.degree[root] == 1 and tree.adjList[root][0].dest == parent:
            for k in range(tree.number_of_sites):
                self.dp[root][k][np.min([tree.kmer_count[root][k], self.max_kmer_cnt - 1])] = 1
            return
        else:

            for e in tree.adjList[root]:
                if e is None or e.id == -1:
                    break
                if e.dest == parent:
                    continue

                v = e.dest
                self.compute_likelihood_helper(tree, v, root)

            self.relax_node(tree, root, parent)

    def relax_edge(self, edge_id, new_length):
        self.tree.update_branch_length(edge_id, new_length)

        src = self.tree.edges[edge_id].src
        dest = self.tree.edges[edge_id].dest

        root = src
        if self.parents[dest] != src:
            root = dest

        while root != -1:
            parent = self.parents[root]
            self.relax_node(self.tree, root, parent)
            root = parent

    def compute_log_likelihood(self, root, recompute_table=True):
        mat = np.zeros(shape=(self.tree.number_of_sites,))

        if recompute_table:
            self.compute_likelihood_helper(self.tree, root, -1)

        mat = np.log(np.matmul(self.dp[root], self.pi)) - self.lg_factor[root]
        ans = np.sum(mat)
        return ans

