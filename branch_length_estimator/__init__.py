import math
import random
import time
from copy import deepcopy

import numpy as np
from scipy import optimize

import phylogenetic_tree
from felsenstein_pruner import gen_censored_linear_bdps_qmat, FelsensteinPruner
from phylogenetic_tree import initialize_tree_with_random_branch_lengths


def estimate_branch_length(tree, lower, upper, max_kmer_count,
                                         num_passes=30, eprecisson=np.float32(0.001),
                                         lprecisson=np.float32(0.0001), seed_tree=False, logfile=None):
    if not seed_tree:
        initialize_tree_with_random_branch_lengths(tree, lower, upper)

    Q = gen_censored_linear_bdps_qmat(tree.lamda, tree.m, tree.mu, max_kmer_count)
    pruner = FelsensteinPruner(tree, qmat=Q)

    it = 0

    prev = -math.inf
    lengths = np.array([e.length for e in tree.edges])
    cntr = 0
    max_cntr = 3

    if logfile is not None:
        phylogenetic_tree.dump_to_file(tree, logfile + "-init.tree")

    while it < num_passes:
        edge_ids = [i for i in range(len(tree.edges))]
        random.shuffle(edge_ids)

        for id in edge_ids:
            e = tree.edges[id]
            start_time = time.time()
            print("iteration: ", it, " edge: ", e.id, "length: ", e.length)
            root = e.src
            if 1 <= root <= tree.number_of_species:
                root = e.dest
            edge_id = e.id
            pruner.compute_log_likelihood(root, recompute_table=True)

            def func(length):
                pruner.relax_edge(edge_id, length)
                return -pruner.compute_log_likelihood(root, recompute_table=False)

            res = optimize.minimize_scalar(func, bounds=(lower, upper), method='bounded')
            pruner.relax_edge(edge_id, res.x)

            curr = -res.fun

            end_time = time.time()
            print("Time: ", end_time - start_time, "Edge Length", tree.edges[e.id].length)

        if logfile is not None:
            phylogenetic_tree.dump_to_file(tree, logfile + "-" + str(it) + ".tree")

        it += 1

        new_lengths = np.array([e.length for e in tree.edges])
        delta = np.max(np.abs(new_lengths - lengths) / lengths)
        print(new_lengths)
        print("Likelihood: ", pruner.compute_log_likelihood(root, recompute_table=False))
        print("Got change ", delta, " at edge ", np.argmax(np.abs(new_lengths - lengths) / lengths))
        if delta < eprecisson:
            print("branch length low changes (converged) at iteration: ", it)
            break
        lengths = new_lengths

        if np.abs(curr - prev) < lprecisson:
            cntr += 1
            if cntr == max_cntr:
                print("Likelihood low changes (converged) at iteration: ", it)
                break
        else:
            cntr = 0

        prev = curr
