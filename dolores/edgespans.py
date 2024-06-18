import tskit
import numpy as np
import scipy
import random
import os
import sys
import gzip
from collections import defaultdict
from tqdm import tqdm
from . import simulations

"""
Here Ne is eff pop size of HAPLOIDS
so Ne = species.population_size * 2
coalescent time = time in generations / Ne
rho/2 = r * Ne
theta/2 = mu * Ne
All times are on the coalescent time scale
"""


class EdgeInfo:
    def __init__(self, num):
        self.name = ""
        self.num = num
        self.id = np.zeros(num, dtype=int)
        self.treeindex = np.zeros(num, dtype=int)
        self.span = np.zeros(num, dtype=float)
        self.time = np.zeros(num, dtype=float)
        self.time_of_B = np.zeros(num, dtype=float)
        self.tree_tbl = np.zeros(num, dtype=float)
        self.depth = np.zeros(num, dtype=int)
        self.cladesize = np.zeros(num, dtype=int)
        self.age = np.zeros(num, dtype=float)
        self.age_norm = np.zeros(num, dtype=float)
        self.prob = np.zeros(num, dtype=float)
        self.q = np.zeros(num, dtype=float)
        self.qsorted = np.zeros(num, dtype=float)

    def compute_age_norm(self):
        self.age_norm = self.age / self.tree_tbl

    def trim(self, i):
        self.num = i
        for attr, value in self.__dict__.items():
            if attr not in ["name", "num"]:
                setattr(self, attr, value[0:i])

    def sortq(self):
        self.qsorted = np.sort(self.q)

    def print_info(self):
        print(self.name, self.num)

    def write_to_file(self, filehandle):
        with gzip.open(filehandle + ".edges.gz", "wt") as file:
            file.write("NAME " + self.name + "\n")
            file.write("NUM_edges " + str(self.num) + "\n")
            for i in range(self.num):
                file.write(
                    str(i)
                    + ";"
                    + str(self.id[i])
                    + ";"
                    + str(self.treeindex[i])
                    + ";"
                    + str(self.span[i])
                    + ";"
                    + str(self.time[i])
                    + ";"
                    + str(self.time_of_B[i])
                    + ";"
                    + str(self.tree_tbl[i])
                    + ";"
                    + str(self.depth[i])
                    + ";"
                    + str(self.cladesize[i])
                    + ";"
                    + str(self.age[i])
                    + ";"
                    + str(self.age_norm[i])
                    + ";"
                    + str(self.prob[i])
                    + ";"
                    + str(self.q[i])
                    + "\n"
                )


class Trees:
    def __init__(self, num_trees):
        self.trees = [None] * num_trees


class Tree:
    def __init__(self, t):
        self.num_lineages = np.zeros(t.num_samples(), dtype=int)
        self.tim_lineages = np.zeros(t.num_samples(), dtype=float)
        self.time_grid = None
        self.time_dict = None
        self.time_dict_rev = None
        self.Ls = np.zeros(t.num_samples(), dtype=float)
        self.ns_table = {}
        self.Ls_table = {}
        self.Qkj_table = {}
        self.tbl = None

    def compute(self, t, epoch_starts=None, Ne_list=None):
        (
            self.num_lineages,
            self.tim_lineages,
            self.time_dict,
            self.time_grid,
            self.time_dict_rev,
        ) = ltt(t, epoch_starts, Ne_list)
        self.Ls = Ls_(self)
        self.tbl = Ls(0, self)

    def lookup_ns(self, T):
        if T in self.ns_table:
            x = self.ns_table[T]
        else:
            x = ns(T, self)
            self.ns_table[T] = x
        return x

    def lookup_Ls(self, T):
        if T in self.Ls_table:
            x = self.Ls_table[T]
        else:
            x = Ls(T, self)
            self.Ls_table[T] = x
        return x

    def lookup_Qkj(self, k, j):
        if (k, j) in self.Qkj_table:
            x = self.Qkj_table[(k, j)]
        else:
            x = Qkj(k, j, self)
            self.Qkj_table[(k, j)] = x
        return x


def read_from_file(filehandle):
    filename = filehandle + ".edges.gz"
    if not os.path.exists(filename):
        sys.exit("Error: no .edges.gz file found.")
    else:
        info = [""] * 2
        with gzip.open(
            filename,
            "rt",
        ) as file:
            for l, line in enumerate(file):
                if l <= 1:
                    line = line.strip().split(" ")
                    print(line)
                    if len(line) > 1:
                        info[l] = line[1]
                else:
                    if l == 2:
                        results = EdgeInfo(float(info[1]))
                        results.name = info[0]
                    line = line.strip().split(";")
                    i = int(line[0])
                    results.id[i] = int(line[1])
                    results.treeindex[i] = int(line[2])
                    results.span[i] = float(line[3])
                    results.time[i] = float(line[4])
                    results.time_of_B[i] = float(line[5])
                    results.tree_tbl[i] = float(line[6])
                    results.depth[i] = int(line[7])
                    results.cladesize[i] = int(line[8])
                    results.age[i] = float(line[9])
                    results.age_norm[i] = float(line[10])
                    results.prob[i] = float(line[11])
                    results.q[i] = float(line[12])
    results.sortq()

    print("Read in", results.num, "edges")

    return results


def Ne_scaling(times, epoch_starts, Ne_list):
    """
    :param epoch_starts: start time of each epoch in generations
    :param Ne_list: should specify the effective population size backwards through time in each epoch
    :param times: unscaled time in generations [T2, ..., Tn, 0]
    """
    if len(epoch_starts) - 1 != len(Ne_list):
        sys.exit(
            "Number of epochs should equal number of piecewise-constant Ne values."
        )
    C = 0
    i = 0
    n = len(times)
    nl = n
    times_ = np.flip(times)
    times_rescaled = np.zeros(len(times), dtype=float)
    time_grid = {n: [[0.0], [Ne_list[0]]]}
    time_dict = {}
    time_dict_rev = {}
    for e1, e2, N in zip(epoch_starts[:-1], epoch_starts[1:], Ne_list):
        while times_[i] <= e2:
            tr = C + (times_[i] - e1) / N
            times_rescaled[i] = tr
            time_dict[times_[i]] = tr
            time_dict_rev[tr] = times_[i]
            nl = n - i
            if n - i > 1:
                time_grid[nl] = [[tr], []]
            if i > 0:
                time_grid[nl + 1][0].append(tr)
                time_grid[nl + 1][1].append(N)
            i += 1
            if i == len(times_):
                break
        C += (e2 - e1) / N
        if nl > 1:
            time_grid[nl][0].append(C)
            time_grid[nl][1].append(N)
        if i == len(times_):
            break
    times_rescaled = [f for f in np.flip(times_rescaled)]
    return times_rescaled, time_dict, time_grid, time_dict_rev


def merge_results(results_set):
    N = 0
    for results in results_set:
        N += results.num

    results_out = EdgeInfo(N)
    results_out.name = results_set[0].name
    results_out.num = N
    results_out.id = np.concatenate([results.id for results in results_set], axis=None)
    results_out.treeindex = np.concatenate(
        [results.treeindex for results in results_set], axis=None
    )
    results_out.span = np.concatenate(
        [results.span for results in results_set], axis=None
    )
    results_out.time = np.concatenate(
        [results.time for results in results_set], axis=None
    )
    results_out.time_of_B = np.concatenate(
        [results.time_of_B for results in results_set], axis=None
    )
    results_out.tree_tbl = np.concatenate(
        [results.tree_tbl for results in results_set], axis=None
    )
    results_out.depth = np.concatenate(
        [results.depth for results in results_set], axis=None
    )
    results_out.cladesize = np.concatenate(
        [results.cladesize for results in results_set], axis=None
    )
    results_out.age = np.concatenate(
        [results.age for results in results_set], axis=None
    )
    results_out.age_norm = np.concatenate(
        [results.age_norm for results in results_set], axis=None
    )
    results_out.prob = np.concatenate(
        [results.prob for results in results_set], axis=None
    )
    results_out.q = np.concatenate([results.q for results in results_set], axis=None)
    results_out.sortq()

    print(len(results_out.id) - len(np.unique(results_out.id)), "duplicated edges")

    return results_out


def compute_trees(
    ts, epoch_starts, Ne_list, sample_trees=None, polytomies=False, quiet=False
):
    """
    Pre-compute variables for each tree in a tree sequence (number of lineages through time,
    event times, L(s) and total edge length)
    :param ts: tree sequence in tskit format
    :param epoch_starts: list of epoch starts (in generations)
    :param Ne_list: population size in each epoch (haploid)
    :param polytomies: whether ts has polytomies
    :param sample_trees: list of trees to compute variables for (only those where at least one edge has been sampled)
    :param quiet: switch off progress bar
    :return:
    """
    trees = Trees(ts.num_trees)
    total_trees = ts.num_trees
    if sample_trees is not None:
        total_trees = len(sample_trees)
    with tqdm(
        total=total_trees,
        desc="Computing tree variables",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=quiet,
    ) as pbar:
        for t in ts.trees():
            do_tree = True
            if sample_trees is not None:
                do_tree = t.index in sample_trees
            if do_tree:
                if t.total_branch_length > 0:
                    if polytomies:
                        tt = t.split_polytomies(random_seed=1)
                    else:
                        tt = t
                    tree = Tree(tt)
                    tree.compute(tt, epoch_starts, Ne_list)
                    trees.trees[t.index] = tree
                pbar.update(1)
    return trees


def ltt(t, epoch_starts, Ne_list, verbose=False):
    """
    Number of lineages between jumps:
    num_lineages = [1, 2, ..., num_samples]
    tim_lineages = [T_2, ..., T_{num_samples}, 0]
    :param t: tree in tskit format
    :param epoch_starts: list of epoch starts (in generations)
    :param Ne_list: population size in each epoch (haploid)
    :param verbose: print info
    :return:
    """
    num_lineages = np.zeros(t.num_samples(), dtype=int)
    tim_lineages = np.zeros(t.num_samples(), dtype=float)
    if verbose:
        print(len(num_lineages))
    count = 1
    for i, n in enumerate(t.nodes(order="timedesc")):
        if t.time(n) == 0:
            if verbose:
                print("Reached samples, recording", count)
            num_lineages[i] = count
            break
        else:
            if verbose:
                print("Recording node", n, "with count", count)
            num_lineages[i] = count
            tim_lineages[i] = t.time(n)
            count += 1
    if verbose:
        print("Finished, count =", count)
    num_lineages = num_lineages[0:count]
    tim_lineages = tim_lineages[0:count]
    tim_lineages, tim_dict, time_grid, tim_dict_rev = Ne_scaling(
        tim_lineages, epoch_starts, Ne_list
    )
    return num_lineages, tim_lineages, tim_dict, time_grid, tim_dict_rev


def ns(s, tree):
    """
    Number of lineages at time s
    :param s: time (coalescent units)
    :param tree: tree object
    :return:
    """
    i = 0
    if s == 0:
        return tree.num_lineages[-1]
    elif s < 0:
        sys.exit("s should be non-negative")
    else:
        while s < tree.tim_lineages[i]:
            i += 1
        return tree.num_lineages[i]


def Ls(s, tree):
    """
    Total edge length of tree above time s
    :param s: time (coalescent units)
    :param tree: tree object
    :return:
    """
    L = 0
    i = 1
    while s <= tree.tim_lineages[i]:
        L += tree.num_lineages[i] * (tree.tim_lineages[i - 1] - tree.tim_lineages[i])
        i += 1
        if i == len(tree.num_lineages):
            break
    if s != 0 and s < tree.tim_lineages[0]:
        L += tree.num_lineages[i] * (tree.tim_lineages[i - 1] - s)
    return L


def Ls_(tree):
    """
    Total edge length of tree above each event time (coalescent units)
    :param tree: tree object
    :return:
    """
    L = np.zeros(len(tree.num_lineages))
    for i in range(len(tree.num_lineages)):
        if i != 0:
            L[i] = L[i - 1] + tree.num_lineages[i] * (
                tree.tim_lineages[i - 1] - tree.tim_lineages[i]
            )
    return L


def ponb(t, b, tree):
    """
    Probability that recombination point is on edge b
    :param t: tree in tskit format
    :param b: edge id
    :param tree: tree object
    :return:
    """
    return (tree.time_dict[t.time(t.parent(b))] - tree.time_dict[t.time(b)]) / tree.tbl


def Q1(k, tree):
    """
    widetilde{Q}^1(k)
    :param k: index k
    :param tree: tree object
    :return:
    """
    return 1 / k * (tree.tim_lineages[k - 2] - tree.tim_lineages[k - 1])


def Qkj(k, j, tree):
    """
    Q_{kj}
    :param k: index k
    :param j: index j
    :param tree: tree object
    :return:
    """
    Tk = tree.tim_lineages[k - 2]
    if k == j:
        Q = -1 / k * np.exp(-k * Tk)
    else:
        Tj = tree.tim_lineages[j - 2]
        Tj1 = tree.tim_lineages[j - 1]
        # LTk = tree.lookup_Ls(Tk)
        LTk = tree.Ls[k - 2]
        # LTj1 = tree.lookup_Ls(Tj1)
        LTj1 = tree.Ls[j - 1]
        Q = 1 / j * (1 - np.exp(-j * (Tj - Tj1))) * np.exp(-k * Tk + LTj1 - LTk)
    return Q


def Q2kxyAB(k, x, y, A, B, tree):
    """
    widetilde{Q}^2(k, x, y, A, B)
    :param k: index k
    :param x: index x
    :param y: index y
    :param A: summation index A
    :param B: summations index B
    :param tree: tree object
    :return:
    """
    Tk = tree.tim_lineages[k - 2]
    Tk1 = tree.tim_lineages[k - 1]
    p = 0
    for j in range(y, x + 1):
        p += (A * j + B) * tree.lookup_Qkj(k, j)
    p = p * 1 / k * (np.exp(k * Tk) - np.exp(k * Tk1))
    return p


def Q2(k, c, b, t, tree):
    """
    Special case: widetilde{Q}^2(k, x, n(t^uparrow(b)) + 1, 0, 1)
    :param k: index k
    :param c: index c
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    y = 1 + tree.lookup_ns(tree.time_dict[t.time(t.parent(b))])
    return Q2kxyAB(k, c, y, 0, 1, tree)


def Q3(k, tree):
    """
    widetilde{Q}^3(k)
    :param k: index k
    :param tree: tree object
    :return:
    """
    LTk = tree.Ls[k - 2]
    LTk1 = tree.Ls[k - 1]
    return 1 / k * (np.exp(-LTk) - np.exp(-LTk1))


def prob_Cnotinb_Rinb(b, t, tree, top_only=False, verbose=False):
    """
    Probability that C notin b given R in b
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :param top_only: whether to only consider topology-changing events
    :param verbose: print info
    :return:
    """
    down = tree.lookup_ns(tree.time_dict[t.time(b)])
    up = tree.lookup_ns(tree.time_dict[t.time(t.parent(b))])
    parup = up
    sib = t.left_sib(b)
    if sib == tskit.NULL:
        sib = t.right_sib(b)
    sibdown = tree.lookup_ns(tree.time_dict[t.time(sib)])
    if top_only:
        if t.parent(b) != t.root:
            parup = tree.lookup_ns(tree.time_dict[t.time(t.parent(t.parent(b)))])
    P = 0
    if verbose:
        print(
            "Edge above node",
            b,
            "\n",
            "tdown =",
            tree.time_dict[t.time(b)],
            "tup =",
            tree.time_dict[t.time(t.parent(b))],
            "\n",
            "index from ",
            down,
            "to",
            up,
            "\n",
        )
    for k in range(up + 1, down + 1):
        if verbose:
            print("summing k =", k)
        if top_only:
            if k <= sibdown:
                P += (
                    2 * Q1(k, tree)
                    + 2 * Q2(k, k, b, t, tree)
                    + Q2kxyAB(k, up, 1 + parup, 0, 1, tree)
                )
            else:
                P += (
                    Q1(k, tree)
                    + Q2kxyAB(k, k, 1 + sibdown, 0, 1, tree)
                    + 2 * Q2kxyAB(k, sibdown, 1 + up, 0, 1, tree)
                    + Q2kxyAB(k, up, 1 + parup, 0, 1, tree)
                )
        else:
            P += Q1(k, tree) + Q2(k, k, b, t, tree)
    P = 1 - P / (tree.time_dict[t.time(t.parent(b))] - tree.time_dict[t.time(b)])
    return P


def get_Bb(b, t):
    """
    mathcal{B}(b)
    :param b: edge id
    :param t: tree in tskit format
    :return:
    """
    b_sib = t.left_sib(b)
    if b_sib == -1:
        b_sib = t.right_sib(b)
    Bb = [b, b_sib]
    if t.num_children(b) > 0:
        Bb.extend(t.children(b))
    return Bb


def prob_Cinb_RnotinBb(b, t, tree, verbose=False):
    """
    Probability C in b given R notin mathcal{B}(b)
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :param verbose: print info
    :return:
    """
    total = 0
    tdown = tree.time_dict[t.time(b)]
    ndown = tree.lookup_ns(tree.time_dict[t.time(b)])
    Bb = get_Bb(b, t)
    if verbose:
        print("Bb =", Bb)
    Bb_times = [t.time(bb) for bb in Bb]
    Bb_times.extend([0, t.time(t.parent(b))])
    Bb_times = np.sort(Bb_times)
    if verbose:
        print("Bb_times =", Bb_times)
    j = 0
    A = [0, 1, 2, 1, 2]
    if t.time(Bb[1]) < t.time(b):
        A[3] += 2

    tbar = 0
    for bb in Bb:
        tbar += tree.time_dict[t.time(t.parent(bb))] - tree.time_dict[t.time(bb)]

    tprev = 0
    for tt in Bb_times[1:]:
        down = tree.lookup_ns(tree.time_dict[tprev])
        up = tree.lookup_ns(tree.time_dict[tt])
        for k in range(up + 1, down + 1):
            Tk = tree.tim_lineages[k - 2]
            Tk1 = tree.tim_lineages[k - 1]
            if tree.time_dict[tt] <= tdown:
                if verbose:
                    print(
                        "lineages:",
                        k,
                        ", from",
                        Tk1,
                        ", to",
                        Tk,
                        ", calculating Q2, Aj =",
                        A[j],
                    )
                total += (k - A[j]) * Q2(k, ndown, b, t, tree)
            else:
                if verbose:
                    print(
                        "lineages:",
                        k,
                        ", from",
                        Tk1,
                        ", to",
                        Tk,
                        ", calculating Q1 + Q2, Aj =",
                        A[j],
                    )
                total += (k - A[j]) * (Q1(k, tree) + Q2(k, k, b, t, tree))
        tprev = tt
        j += 1

    total = total / (tree.tbl - tbar)
    return total


def prob_b_disrupted(b, t, tree, top_only=False):
    """
    Probability edge b is disrupted by the recombination event
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :param top_only: whether to consider only topology-changing events
    :return:
    """
    total = 0
    Bb = get_Bb(b, t)
    p1 = 0
    for bb in Bb:
        total += ponb(t, bb, tree) * prob_Cnotinb_Rinb(bb, t, tree, top_only)
        p1 += tree.time_dict[t.time(t.parent(bb))] - tree.time_dict[t.time(bb)]
    p1 = 1 - p1 / tree.tbl
    total += p1 * prob_Cinb_RnotinBb(b, t, tree)
    return total


def check_cwr_edges(ts):
    """
    Check for ancestral bridges in a CwR tree sequence
    :param ts: tskit tree sequence
    :return:
    """
    edge_dict = defaultdict(list)
    num_bridges = 0
    print("Checking CwR trees for bridges")
    with tqdm(
        total=ts.num_edges, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    ) as pbar:
        for e in ts.edges():
            for ee in ts.edges():
                if ee.id > e.id and e.child == ee.child and e.parent == ee.parent:
                    edge_dict[e.id].append(ee.id)
            if e.id in edge_dict:
                num_bridges += 1
            pbar.update(1)
    print("Number of bridged edges:", num_bridges)
    return edge_dict


def test_tree_effect(
    ts,
    epoch_starts,
    Ne_list,
    sample_edges=None,
):
    """
    Compute probability of edge disruption in each tree where the edge exists
    :param ts:
    :param epoch_starts:
    :param Ne_list:
    :param sample_edges:
    :return:
    """
    total_edges = ts.num_edges
    if sample_edges is not None:
        if sample_edges > ts.num_edges:
            sample_edges = ts.num_edges
        sample_edges = random.sample(range(ts.num_edges), sample_edges)
        sample_edges.sort()
        sample_edges = set(sample_edges)
        total_edges = len(sample_edges)
    trees = compute_trees(ts, epoch_starts, Ne_list, sample_trees=None)

    results_all = []
    results_sum = [None] * len(sample_edges)
    i = 0

    with tqdm(total=total_edges, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
        for e in ts.edges():
            if e.id in sample_edges:
                Pset = []
                Pprev = -1
                t = ts.at(e.left)
                assert e.left == t.interval[0]
                while t.interval[1] <= e.right:
                    if t.num_roots == 1:
                        tree = trees.trees[t.index]
                        if not t.is_descendant(e.child, e.parent):
                            print(e.id, e.left, e.right)
                            sys.exit(
                                "Node "
                                + str(e.child)
                                + " not a descendant of "
                                + str(e.parent)
                                + " in tree "
                                + str(t.index)
                            )
                        P = prob_b_disrupted(e.child, t, tree) * t.total_branch_length
                        Pset.append(P)
                        if Pprev != -1:
                            results_all.append(P / Pprev)
                        Pprev = P
                    else:
                        print("Number of roots in tree " + str(t.index) + " > 1")
                    if not t.next():
                        break
                results_sum[i] = max(Pset) / min(Pset)
                i += 1
                pbar.update(1)

    return results_all, results_sum


def record_results(
    results,
    i,
    e,
    t,
    tree,
    ts,
    rec_map,
    top_only,
    L=None,
    Left=None,
    Right=None,
    supported_only=False,
    mutation_rate=None,
):
    if L is None:
        L = (Right - Left) / ts.sequence_length
    P = prob_b_disrupted(e.child, t, tree, top_only)
    if Right > rec_map.sequence_length:
        print(
            "Warning: right endpoint",
            Right,
            "> sequence length",
            rec_map.sequence_length,
        )
        Right = rec_map.sequence_length
    R = rec_map.get_cumulative_mass(Right) - rec_map.get_cumulative_mass(Left)

    if P <= 0:
        sys.exit("Error: probability cannot be < 0")

    results.id[i] = e.id
    results.treeindex[i] = t.index
    results.span[i] = L
    results.tree_tbl[i] = tree.tbl
    results.prob[i] = P
    results.time[i] = tree.time_dict[t.time(e.parent)] - tree.time_dict[t.time(e.child)]
    results.time_of_B[i] = (
        tree.time_dict[t.time(e.parent)] - tree.time_dict[t.time(e.child)]
    )
    results.depth[i] = t.depth(e.parent)
    for n in [
        t.right_sib(e.child),
        t.left_sib(e.child),
        t.right_child(e.child),
        t.left_child(e.child),
    ]:
        if n != tskit.NULL:
            results.time_of_B[i] += (
                tree.time_dict[t.time(t.parent(n))] - tree.time_dict[t.time(n)]
            )
    results.age[i] = tree.time_dict[t.time(e.parent)]
    results.cladesize[i] = t.num_samples(e.child)

    if supported_only:
        d = e.right - e.left
        lamb = P * t.total_branch_length * R
        thet = (
            mutation_rate
            * (tree.time_dict[t.time(e.parent)] - tree.time_dict[t.time(e.child)])
            * d
        )
        results.q[i] = (
            1
            - (thet + lamb) / thet * np.exp(-lamb)
            + lamb / thet * np.exp(-(lamb + thet))
        )
    else:
        results.q[i] = scipy.stats.expon.cdf(
            P * t.total_branch_length * R, loc=0, scale=1
        )

    if results.q[i] == 0 and R != 0:
        print(P, t.total_branch_length, R)
        sys.exit("Error: q = 0.")


def gof_calc(
    ts,
    rec_map,
    epoch_starts,
    Ne_list,
    top_only=False,
    supported_only=False,
    mutation_rate=None,
    tsdate_trees=False,
    relate_trees=False,
    relate_anc_file=None,
    argw_trees=False,
    argn_trees=False,
    argn_norm=False,
    cwr_trees=False,
    num_sample_edges=None,
    sample_range=None,
    pos_range=None,
    name=None,
    write_to_file=None,
):
    """
    Calculation of widetilde{q}_i for a given ARG
    :param num_sample_edges: number of edges to sample from the ARG
    :param sample_range: (start, end) tuple giving range of edge ids to sample from - not for Relate trees. Left inclusive, right exclusive.
    :param pos_range: (start, end) range of positions - for Relate trees
    :param ts: tree sequence in tskit format
    :param rec_map: recombination map object (msprime style)
    :param epoch_starts: list of epochs with piecewise constant population sizes (generations)
    :param Ne_list: list of effective population size in each epoch (haploid)
    :param top_only: whether to consider only topology-changing events
    :param supported_only: whether to only test edges than have at least one mutation
    :param mutation_rate: mutation rate per bp per generation
    :param tsdate_trees: whether ts is from tsdate
    :param relate_trees: whether ts is from Relate
    :param relate_anc_file: path to Relate anc file (to compute edge spans)
    :param argw_trees: whether ts is from ARGweaver
    :param argn_trees: whether ts is from ARG-needle
    :param argn_norm: whether ts is from ARG-needle with normalisation (in this case it's in coalescent time units, so won't scale by 2*Ne)
    :param cwr_trees: whether ts is from CwR simulation
    :param name: label for results (if not provided, will be inferred from ts)
    :param write_to_file: file handle for storing results (no extension)
    :return:
    """

    results = EdgeInfo(ts.num_edges)
    t = ts.first()
    i = 0

    if supported_only:
        if mutation_rate is None:
            sys.exit("Need to provide mutation rate if only using supported edges.")

    if name:
        results.name = name
    else:
        if relate_trees:
            results.name = "Relate"
        elif tsdate_trees:
            results.name = "tsdate"
        elif argw_trees:
            results.name = "ARGweaver"
        elif argn_trees:
            results.name = "ARG-needle"
        elif cwr_trees:
            results.name = "CwR"
        else:
            results.name = "Simulation"

    if relate_trees:
        if pos_range is not None:
            edges, edge_lengths_all, _ = simulations.read_anc_file(
                relate_anc_file,
                ts.sequence_length,
                supported_only=supported_only,
                pos_range=pos_range,
            )
        else:
            edges, edge_lengths_all, _ = simulations.read_anc_file(
                relate_anc_file,
                ts.sequence_length,
                supported_only=supported_only,
            )

    offset = 0
    if argn_trees and len(ts.metadata) != 0:
        offset = int(ts.metadata["offset"]) - 1
        print("Adding offset", offset, "to all edge coordinates")

    sample_trees = None
    if num_sample_edges is not None:
        if relate_trees:
            if num_sample_edges > ts.num_edges:
                num_sample_edges = len(edges)
            sample_edges = set(
                random.sample({v for v in edges.values()}, num_sample_edges)
            )
            sample_trees = {t for (t, _) in sample_edges}
        else:
            if sample_range is None:
                sample_range = (0, ts.num_edges)
            if supported_only:
                sample_range = {
                    m.edge
                    for m in ts.mutations()
                    if sample_range[0] <= m.edge < sample_range[1]
                }
            else:
                sample_range = range(sample_range[0], sample_range[1])
            if num_sample_edges > len(sample_range):
                num_sample_edges = len(sample_range)
            sample_edges = random.sample(sample_range, num_sample_edges)
            sample_edges.sort()
            sample_trees = set()
            with tqdm(
                total=len(sample_edges),
                desc="Computing tree variables",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ) as pbar:
                for edge in sample_edges:
                    e = ts.edge(edge)
                    t.seek(e.left)
                    sample_trees.add(t.index)
                    pbar.update(1)
            sample_edges = set(sample_edges)
    else:
        if relate_trees:
            sample_edges = {v for v in edges.values()}
        else:
            if supported_only:
                sample_edges = {m.edge for m in ts.mutations()}
            else:
                sample_edges = range(ts.num_edges)
        num_sample_edges = len(sample_edges)

    if tsdate_trees:
        trees = compute_trees(ts, epoch_starts, Ne_list, sample_trees, polytomies=True)
    else:
        trees = compute_trees(ts, epoch_starts, Ne_list, sample_trees)

    if cwr_trees:
        edge_track = np.zeros(ts.num_edges, dtype=int)
        edge_dict = check_cwr_edges(ts)

        with tqdm(
            total=num_sample_edges,
            desc="Computing edge p-values",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as pbar:
            for e in ts.edges():
                if edge_track[e.id] == 0 and e.id in sample_edges:
                    t.seek(e.left)
                    tree = trees.trees[t.index]

                    L = (e.right - e.left) / ts.sequence_length
                    if e.id in edge_dict:
                        for ee_id in edge_dict[e.id]:
                            ee = ts.edge(ee_id)
                            L += (ee.right - ee.left) / ts.sequence_length
                            edge_track[ee_id] = 1

                    record_results(
                        results,
                        i,
                        e,
                        t,
                        tree,
                        ts,
                        rec_map,
                        top_only,
                        L=L,
                        supported_only=supported_only,
                        mutation_rate=mutation_rate,
                    )
                    i += 1
                pbar.update(1)
    elif relate_trees:
        with tqdm(
            total=num_sample_edges,
            desc="Computing edge p-values",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as pbar:
            if sample_edges is not None:
                for treeindex, edgeindex in sample_edges:
                    t = ts.at_index(treeindex)
                    tree = trees.trees[t.index]
                    nodeindex = edgeindex
                    if edgeindex >= ts.num_samples:
                        nodeindex = edgeindex + (ts.num_samples - 1) * treeindex
                    e = ts.edge(t.edge(nodeindex))
                    Left, Right = edge_lengths_all[t.index][edgeindex]
                    if Left != -1:
                        record_results(
                            results,
                            i,
                            e,
                            t,
                            tree,
                            ts,
                            rec_map,
                            top_only,
                            Left=Left,
                            Right=Right,
                            supported_only=supported_only,
                            mutation_rate=mutation_rate,
                        )
                        i += 1
                    pbar.update(1)
            else:
                for e in ts.edges():
                    t.seek(e.left)
                    tree = trees.trees[t.index]
                    ind = e.child
                    if not t.is_sample(e.child):
                        ind = ts.num_samples + int(
                            (ind - ts.num_samples) % (ts.num_samples - 1)
                        )
                    Left, Right = edge_lengths_all[t.index][ind]
                    if Left != -1:
                        record_results(
                            results,
                            i,
                            e,
                            t,
                            tree,
                            ts,
                            rec_map,
                            top_only,
                            Left=Left,
                            Right=Right,
                            supported_only=supported_only,
                            mutation_rate=mutation_rate,
                        )
                        i += 1
                    pbar.update(1)
    else:
        with tqdm(
            total=num_sample_edges,
            desc="Computing edge p-values",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as pbar:
            for e in ts.edges():
                if e.id in sample_edges:
                    t.seek(e.left)
                    tree = trees.trees[t.index]

                    if tsdate_trees:
                        tt = t.split_polytomies(random_seed=1)
                    else:
                        tt = t

                    record_results(
                        results,
                        i,
                        e,
                        tt,
                        tree,
                        ts,
                        rec_map,
                        top_only,
                        Left=e.left + offset,
                        Right=e.right + offset,
                        supported_only=supported_only,
                        mutation_rate=mutation_rate,
                    )
                    i += 1
                    pbar.update(1)

    if top_only:
        results.name = results.name + " (top_ch)"
    results.trim(i)
    results.sortq()
    results.compute_age_norm()

    if write_to_file is not None:
        results.write_to_file(write_to_file)

    return results
