import io

import numpy as np
from Bio import Phylo
from dataclasses import dataclass

@dataclass
class Edge:
    id: int = -1
    src: int = -1
    dest: int = -1
    length: np.float32 = -1
    

class PhylogeneticTree:
    def __init__(self, number_of_species, number_of_sites, lamda, m, mu, max_degree_of_a_node=3):
        self.lamda = lamda
        self.mu = mu
        self.m = m
        self.number_of_sites = number_of_sites
        self.n = number_of_species * 2 - 1
        self.number_of_species = number_of_species

        self.node_names = [str(i) for i in range(self.n)]
        self.edges = []
        self.kmer_count = np.zeros(shape=(self.n, number_of_sites), dtype=np.int32)

        self.adjList = np.ndarray(shape=(self.n, max_degree_of_a_node + 1), dtype=Edge)
        self.degree = np.zeros(shape=(self.n,), dtype=np.int32)

    def add_edge(self, u, v, w):
        new_edge = Edge(len(self.edges), u, v, w)
        self.edges.append(new_edge)
        self.adjList[u][self.degree[u]] = new_edge
        self.degree[u] += 1
        new_edge = Edge(new_edge.id, v, u, w)
        self.adjList[v][self.degree[v]] = new_edge
        self.degree[v] += 1

    def update_branch_length(self, edge_id, new_length):
        self.edges[edge_id].length = new_length
        src = self.edges[edge_id].src
        dest = self.edges[edge_id].dest

        for e in self.adjList[src]:
            if e is None or e.id == -1:
                break
            if e.id == edge_id:
                e.length = new_length
                break
        for e in self.adjList[dest]:
            if e is None or e.id == -1:
                break
            if e.id == edge_id:
                e.length = new_length

    def get_newick(self):
        visited = [False] * self.n

        def dfs(tree, root, parent):
            visited[root] = True
            l = []
            for e in tree.adjList[root]:
                if e is None or e.id == -1:
                    break
                if not visited[e.dest]:
                    l.append(dfs(tree, e.dest, root) + ":" + str(e.length))
            s = str(root)
            if len(l) >= 1:
                s = "(" + ','.join(l) + ")" + str(root)
            return s

        return dfs(self, 0, None)
    
    def get_ete3_newick(self):
        visited = [False] * self.n

        def dfs(tree, root, parent):
            visited[root] = True
            l = []
            for e in tree.adjList[root]:
                if e is None or e.id == -1:
                    break
                if not visited[e.dest]:
                    l.append(dfs(tree, e.dest, root) + ":" + str(e.length))
            s = str(root)
            if len(l) >= 1:
                s = "(" + ','.join(l) + ")"
            return s

        return dfs(self, 0, None)

    def get_normalized_newick(self):
        visited = [False] * self.n
        f = np.sum(np.array([e.length for e in self.edges]))

        def dfs(tree, root, parent):
            visited[root] = True
            l = []
            for e in tree.adjList[root]:
                if (e is None or e.id == -1):
                    break
                if (not visited[e.dest]):
                    l.append(dfs(tree, e.dest, root) + ":" + str(e.length / f))
            s = str(root)
            if len(l) >= 1:
                s = "(" + ','.join(l) + ")" + str(root)
            return s

        return dfs(self, 0, None)

    def __repr__(self):
        p = "lamda: {},\
        \n\tmu: {},\
        \n\tm: {},\
        \n\tnum_species: {},\
        \n\tnum_independent_sites: {},\
        \n\tadjList: {}\t,\
        \n\tkmer_count: {}\n"

        adjList = "\n"
        for i in range(self.n):
            for e in self.adjList[i]:
                if e is None or e.id == -1:
                    break
                adjList += "\t\t" + str(e) + "\n"

        site_cnt = "\n"
        for i in range(self.n):
            site_cnt += "\t\t" + str(i) + ': ' + str(self.kmer_count[i]) + '\n'

        return "{\n\t" + p.format(self.lamda, self.mu, self.m,
                                  self.number_of_species, self.number_of_sites,
                                  adjList, site_cnt) + "}"


def load_from_file(filename, only_leaves):
    file = open(filename, 'r')
    num_species = int(file.readline())
    num_sites = int(file.readline())
    lamda, m, mu = file.readline()[:-1].split(' ')

    tree = PhylogeneticTree(number_of_species=num_species, number_of_sites=num_sites,
                            lamda=float(lamda), m=float(m), mu=float(mu))

    for i in range(tree.n - 1):
        s = file.readline()[:-1].split(' ')
        u = s[0]
        v = s[1]
        w = s[2]
        u = np.int32(u)
        v = np.int32(v)
        w = np.float32(w)

        tree.add_edge(u, v, w)

    start = 0
    finish = tree.n
    if only_leaves:
        start = 1
        finish = tree.number_of_species + 1
    for i in range(start, finish):
        lst = file.readline().strip().split(' ')
        # print(len(lst))
        tree.kmer_count[i][:] = lst[:]

    file.close()
    return tree


def dump_to_file(tree, filename):
    file = open(filename, 'w')

    file.write(str(tree.number_of_species) + '\n')
    file.write(str(tree.number_of_sites) + '\n')
    file.write(str(tree.lamda) + ' ' + str(tree.m) + ' ' + str(tree.mu) + '\n')

    for i in range(tree.n - 1):
        file.write(str(tree.edges[i].src) + ' ' + str(tree.edges[i].dest) + ' ' + str(tree.edges[i].length) + '\n')

    for i in range(tree.n):
        for j in range(tree.number_of_sites):
            file.write(str(tree.kmer_count[i][j]) + ' ')
        file.write('\n')

    file.close()


def initialize_tree_with_random_branch_lengths(tree, low, high):
    for e in tree.edges:
        e.length = np.random.uniform(low, high)

    for i in range(tree.n):
        for e in tree.adjList[i]:
            if e is None or e.id == -1:
                break
            e.length = tree.edges[e.id].length


def draw_tree(tree, ax=None):
    treedata = tree.get_normalized_newick()
    handle = io.StringIO(treedata)
    phylo_tree = Phylo.read(handle, "newick")
    Phylo.draw(phylo_tree, axes=ax, do_show=False)

"""
Takes a rooted tree as input
Returns an array of distances of all nodes from the root
"""
def get_dist(root, tree):
    dist = np.ndarray(shape=(tree.n,), dtype=np.float32)
    visited = np.zeros(shape=(tree.n,), dtype=np.int32)

    queue = [root]
    visited[root] = 1
    dist[root] = 0

    while len(queue):
        t = queue[0]
        queue.pop(0)
        for e in tree.adjList[t]:
            if e is None or e.id == -1:
                break
            if visited[e.dest] != 1:
                visited[e.dest] = 1
                dist[e.dest] = dist[e.src] + e.length
                queue.append(e.dest)

    return dist

"""
Takes two trees (with same topology) as input 
and returns the maximum relative between two 
leaf nodes
"""
def get_max_root_to_leaf_error(tree_original, tree_estimate):
    d1 = get_dist(0, tree_original)
    d2 = get_dist(0, tree_estimate)

    error = np.abs(d2 - d1) / d1 * 100
    return np.max(error[1:tree_original.number_of_species])


"""
  standard deviation from 1 of ratio of normalized original and estimate arrays
"""
def get_standard_deviation_branch_length_ratio(tree_original, tree_estimate):
    original = np.array([e.length for e in tree_original.edges])
    estimate = np.array([e.length for e in tree_estimate.edges])

    ratio = np.array([e / o for e, o in zip(estimate, original)])
    ratio = ratio / np.linalg.norm(ratio)

    return np.std(ratio)


def bldk(tree_original, tree_estimate, K):
    return np.sqrt(np.sum((np.array([e.length for e in tree_original.edges]) -
                           K * np.array([e.length for e in tree_estimate.edges])) ** 2)
                   /len(tree_original.edges))

"""
find the value of K for which BLDK is minimum between two trees
"""
def find_optimal_K(tree_original, tree_estimate):
    e_original = np.array([e.length for e in tree_original.edges])
    e_estimate = np.array([e.length for e in tree_estimate.edges])
    return np.sum(e_original * e_estimate) / np.sum(e_estimate * e_estimate)
