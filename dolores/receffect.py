import sys
import msprime
import numpy as np
import random
import math
from tqdm import tqdm
from . import edgespans


def P1kl(kk, ll, tree):
    """
    P^1_{kl}
    :param kk: k index
    :param ll: l index
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    Tk = tree.tim_lineages[kk - 2] / N
    Tl = tree.tim_lineages[ll - 2] / N
    LTk = tree.lookup_Ls(Tk)
    LTl = tree.lookup_Ls(Tl)
    return np.exp(ll * Tl - kk * Tk + LTl - LTk)


def P2k(k, b, t, tree):
    """
    P^2_k
    :param k: k index
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    tup = t.time(t.parent(b)) / N
    Tk = tree.tim_lineages[k - 2] / N
    LTk = tree.lookup_Ls(Tk)
    Ltup = tree.lookup_Ls(tup)
    return np.exp(tup - k * Tk + Ltup - LTk)


def P3k(k, b, t, tree):
    """
    P^3_k
    :param k: k index
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    tup = t.time(t.parent(b)) / N
    Tk = tree.tim_lineages[k - 2] / N
    LTk = tree.lookup_Ls(Tk)
    return np.exp(tup - k * Tk - LTk)


def Ptilde1kl(kk, ll, tree):
    """
    widetilde{P}^1_{kl}
    :param kk: k index
    :param ll: l index
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    Tk = tree.tim_lineages[kk - 2] / N
    Tk1 = tree.tim_lineages[kk - 1] / N
    return (
        1
        / (kk - 1)
        * (np.exp((kk - 1) * Tk) - np.exp((kk - 1) * Tk1))
        * P1kl(kk, ll, tree)
    )


def Ptilde2k(k, b, t, tree):
    """
    widetilde{P}^2_k
    :param k: k index
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    Tk = tree.tim_lineages[k - 2] / N
    Tk1 = tree.tim_lineages[k - 1] / N
    return (
        1
        / (k - 1)
        * (np.exp((k - 1) * Tk) - np.exp((k - 1) * Tk1))
        * P2k(k, b, t, tree)
    )


def Ptilde3k(k, b, t, tree):
    """
    widetilde{P}^3_k
    :param k: k index
    :param b: edge id
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    Tk = tree.tim_lineages[k - 2] / N
    Tk1 = tree.tim_lineages[k - 1] / N
    return (
        1
        / (k - 1)
        * (np.exp((k - 1) * Tk) - np.exp((k - 1) * Tk1))
        * P3k(k, b, t, tree)
    )


def pCc_conditional(c_range, b, s, t, tree):
    """
    p^C_mathcal{T}(c | C neq 0, mathcal{R} = (b, s))
    :param c_range: array of c values to calculate the density for
    :param b: edge id of the recombination event
    :param s: time of the recombination event
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    results = np.zeros(len(c_range), dtype=float)
    T2 = tree.tim_lineages[0] / N
    T3 = tree.tim_lineages[1] / N
    tup = t.time(t.parent(b)) / N
    kk = np.where(tree.tim_lineages / N >= s)[0][-1] + 2
    tup_root = t.time(t.parent(b)) == t.time(t.root)
    for indc, c in enumerate(c_range):
        if c == 0:
            results[indc] = None
        elif (not tup_root and c < s - tup) or (tup_root and c < s + T3 - 2 * T2):
            results[indc] = 0
        else:
            if c < 0:
                if not tup_root:
                    ll = np.where(tree.tim_lineages / N >= c + tup)[0][-1] + 2
                    nn = tree.lookup_ns(c + tup)
                    P = (
                        np.exp((kk - 1) * s)
                        * P1kl(kk, ll, tree)
                        * (nn - 1)
                        * np.exp(-(ll - 1) * (c + tup))
                    )
                else:
                    if c <= 2 * (T3 - T2):
                        ll = (
                            np.where(tree.tim_lineages / N >= c - T3 + 2 * T2)[0][-1]
                            + 2
                        )
                        nn = tree.lookup_ns(c - T3 + 2 * T2)
                        P = (
                            np.exp((kk - 1) * s)
                            * P1kl(kk, ll, tree)
                            * (nn - 1)
                            * np.exp(-(ll - 1) * (c - T3 + 2 * T2))
                        )
                    else:
                        ll = np.where(tree.tim_lineages / N >= c / 2 + T2)[0][-1] + 2
                        nn = tree.lookup_ns(c / 2 + T2)
                        P = (
                            0.5 * np.exp((kk - 1) * s)
                            * P1kl(kk, ll, tree)
                            * (nn - 1)
                            * np.exp(-(ll - 1) * (c / 2 + T2))
                        )

            elif c < T2 - tup:
                Ltup = tree.lookup_Ls(tup)
                Lctup = tree.lookup_Ls(c + tup)
                PP = P2k(kk, b, t, tree)
                P = (
                    np.exp((kk - 1) * s)
                    * PP
                    * tree.lookup_ns(c + tup)
                    * np.exp(-(Ltup - Lctup))
                )
            else:
                PP = P3k(kk, b, t, tree)
                P = 0.5 * np.exp((kk - 1) * s) * PP * np.exp(-(c + tup - T2) / 2)
            results[indc] = P
    return results


def pCc(c_range, t, tree, verbose=False):
    """
    p^C_mathcal{T}(c | C != 0)
    :param c_range: array of c values to calculate the density for
    :param t: tree in tskit format
    :param tree: tree object
    :param verbose: print info
    :return:
    """
    N = 2 * tree.Ne
    results = np.zeros((2 * t.num_samples() - 2, len(c_range)), dtype=float)
    T2 = tree.tim_lineages[0] / N
    T3 = tree.tim_lineages[1] / N
    indb = 0
    with tqdm(
        total=(2 * t.num_samples() - 1), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    ) as pbar:
        for b in t.nodes():
            if b != t.root:
                if verbose:
                    print("-" * 15)
                if verbose:
                    print("edge", b)
                tup_root = t.time(t.parent(b)) == t.time(t.root)
                tup = t.time(t.parent(b)) / N
                tdown = t.time(b) / N
                nup = tree.lookup_ns(tup)
                ndown = tree.lookup_ns(tdown)
                if verbose:
                    print(
                        "tup =",
                        tup,
                        ", tdown =",
                        tdown,
                        ", nup =",
                        nup,
                        ", ndown =",
                        ndown,
                    )
                for indc, c in enumerate(c_range):
                    if c == 0:
                        if verbose:
                            print("c = 0")
                        results[indb, indc] = None
                    elif (not tup_root and c < tdown - tup) or (
                        tup_root and c < tdown + T3 - 2 * T2
                    ):
                        if verbose:
                            print("c < minimum")
                        results[indb, indc] = 0
                    else:
                        P = 0
                        if c < 0:
                            if not tup_root:
                                if verbose:
                                    print("c < 0, tup != T2, c =", c)
                                ll = (
                                    np.where(tree.tim_lineages / N > c + tup)[0][-1] + 2
                                )
                                if verbose:
                                    print("l =", ll)
                                Tl = tree.tim_lineages[ll - 2] / N
                                for kk in range(ll, ndown + 1):
                                    if verbose:
                                        print("k =", kk, ", calculating Ptilde1k")
                                    PP = Ptilde1kl(kk, ll, tree)
                                    P += PP
                                P = P * np.exp(-(ll - 1) * (c + tup))
                                P = P - (np.exp((ll - 1) * (Tl - tup - c)) - 1) / (
                                    ll - 1
                                )
                                P = P * (tree.lookup_ns(c + tup) - 1)
                            else:
                                if verbose:
                                    print("c < 0, tup = T2, c =", c)
                                if c < 2 * (T3 - T2):
                                    ll = (
                                        np.where(
                                            tree.tim_lineages / N > c - T3 + 2 * T2
                                        )[0][-1]
                                        + 2
                                    )
                                    if verbose:
                                        print("l =", ll)
                                    Tl = tree.tim_lineages[ll - 2] / N
                                    for kk in range(ll, ndown + 1):
                                        if verbose:
                                            print("k =", kk, ", calculating Ptilde1k")
                                        PP = Ptilde1kl(kk, ll, tree)
                                        P += PP
                                    P = P * np.exp(-(ll - 1) * (c - T3 + 2 * T2))
                                    P = P - (
                                        np.exp((ll - 1) * (Tl - c + T3 - 2 * T2)) - 1
                                    ) / (ll - 1)
                                    P = P * (tree.lookup_ns(c - T3 + 2 * T2) - 1)
                                else:
                                    ll = (
                                        np.where(tree.tim_lineages / N > c / 2 + T2)[0][
                                            -1
                                        ]
                                        + 2
                                    )
                                    if verbose:
                                        print("l =", ll)
                                    if ll != 2:
                                        sys.exit("l should be 2")
                                    Tl = tree.tim_lineages[ll - 2] / N
                                    for kk in range(ll, ndown + 1):
                                        if verbose:
                                            print("k =", kk, ", calculating Ptilde1k")
                                        PP = Ptilde1kl(kk, ll, tree)
                                        P += PP
                                    P = P * np.exp(- (c/2 + T2))
                                    P = P - np.exp(-c/2) + 1
                                    P = P * 0.5 * (tree.lookup_ns(c / 2 + T2) - 1)
                        elif c < T2 - tup:
                            if verbose:
                                print("c < T2 - tup, c =", c)
                            for k in range(nup + 1, ndown + 1):
                                if verbose:
                                    print("k =", k, ", calculating Ptilde2k")
                                P += Ptilde2k(k, b, t, tree)
                            Ltup = tree.lookup_Ls(tup)
                            Lctup = tree.lookup_Ls(c + tup)
                            P = P * tree.lookup_ns(c + tup) * np.exp(-(Ltup - Lctup))
                        else:
                            if verbose:
                                print("c > T2 - tup, c =", c)
                            for k in range(nup + 1, ndown + 1):
                                if verbose:
                                    print("k =", k, ", calculating Ptilde3k")
                                P += Ptilde3k(k, b, t, tree)
                            P = P * 0.5 * np.exp(-(c + tup - T2) / 2)
                        results[indb, indc] = P
                indb += 1
            pbar.update(1)
    return np.sum(results, axis=0) / tree.tbl


def pDzero(t, tree):
    """
    Probability that the tree height does not change
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    P = 0
    for b in t.nodes():
        if b != t.root:
            tup = t.time(t.parent(b)) / N
            tdown = t.time(b) / N
            nup = tree.lookup_ns(tup)
            ndown = tree.lookup_ns(tdown)
            if t.parent(b) == t.root:
                for k in range(nup + 1, ndown + 1):
                    P += edgespans.Q1(k, tree) + edgespans.Q2(k, k, b, t, tree)
            else:
                P += tup - tdown
                for k in range(nup + 1, ndown + 1):
                    P -= edgespans.Q3(k, tree)
    return P / tree.tbl


def pDnegative(t, tree):
    """
    Probability that the tree height decreases
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    P = 0
    for b in t.children(t.root):
        tdown = t.time(b) / N
        ndown = tree.lookup_ns(tdown)
        for k in range(2, ndown + 1):
            P += (
                (k - 1) * edgespans.Q1(k, tree)
                - edgespans.Q2(k, k, b, t, tree)
                - edgespans.Q3(k, tree)
            )
    return P / tree.tbl


def pDpositive(t, tree):
    """
    Probability that the tree height increases
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    P = 0
    for b in t.nodes():
        if b != t.root:
            tup = t.time(t.parent(b)) / N
            tdown = t.time(b) / N
            nup = tree.lookup_ns(tup)
            ndown = tree.lookup_ns(tdown)
            for k in range(nup + 1, ndown + 1):
                P += edgespans.Q3(k, tree)
    return P / tree.tbl


def pCnegative(t, tree):
    """
    Probability that the total edge length decreases
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    P = 0
    for b in t.nodes():
        if b != t.root:
            tup = t.time(t.parent(b)) / N
            tdown = t.time(b) / N
            nup = tree.lookup_ns(tup)
            ndown = tree.lookup_ns(tdown)
            for k in range(nup + 1, ndown + 1):
                P += (k - 1) * edgespans.Q1(k, tree) + edgespans.Q2kxyAB(
                    k, k, nup + 1, 1, -1, tree
                )
    return P / tree.tbl


def pCzero(t, tree):
    """
    Probability that the total edge length does not change
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    P = 0
    for b in t.nodes():
        if b != t.root:
            tup = t.time(t.parent(b)) / N
            tdown = t.time(b) / N
            nup = tree.lookup_ns(tup)
            ndown = tree.lookup_ns(tdown)
            for k in range(nup + 1, ndown + 1):
                P += edgespans.Q1(k, tree) + edgespans.Q2kxyAB(
                    k, k, nup + 1, 0, 1, tree
                )
    return P / tree.tbl


def pCpositive(t, tree):
    """
    Probability that the total edge length increases
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    N = 2 * tree.Ne
    P = 0
    for b in t.nodes():
        if b != t.root:
            tup = t.time(t.parent(b)) / N
            tdown = t.time(b) / N
            nup = tree.lookup_ns(tup)
            ndown = tree.lookup_ns(tdown)
            for k in range(nup + 1, ndown + 1):
                P += edgespans.Q2kxyAB(k, nup, 2, 1, 0, tree) + edgespans.Q3(k, tree)
    return P / tree.tbl


def mean_change(ts, Ne, sim_type="height"):
    """
    Calculate the mean change in the statistic of interest (height or total edge length)
    :param ts: tree sequence in tskit format
    :param Ne: effective population size (diploid)
    :param sim_type: "height" or "tbl"
    :return:
    """
    results_ranked = np.zeros((4, 100), dtype=float)
    with tqdm(
        total=ts.num_trees, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    ) as pbar:
        for t in ts.trees():
            if t.index == 0:
                if sim_type == "height":
                    prevTT = math.floor(t.time(t.root) / (2 * Ne))
                    prevSTAT = t.time(t.root)
                else:
                    prevTT = math.floor(t.total_branch_length / (2 * Ne))
                    prevSTAT = t.total_branch_length
            else:
                if sim_type == "height":
                    STAT = t.time(t.root)
                else:
                    STAT = t.total_branch_length
                results_ranked[0, prevTT] += 1
                if STAT < prevSTAT:
                    results_ranked[1, prevTT] += 1  # decrease
                elif STAT == prevSTAT:
                    results_ranked[2, prevTT] += 1  # zero
                else:
                    results_ranked[3, prevTT] += 1  # increase
                if sim_type == "height":
                    prevTT = math.floor(t.time(t.root) / (2 * Ne))
                else:
                    prevTT = math.floor(t.total_branch_length / (2 * Ne))
                prevSTAT = STAT
            pbar.update(1)

    results_ranked[1:] = results_ranked[1:] / results_ranked[0]
    np.nan_to_num(results_ranked, copy=False)

    return results_ranked


def mean_change_sim(
    sim_type="height",
    n=5,  # diploids
    n_per_bin=10,
    bins_start=0,
    bins_end=10,
):
    """
    Simulate trees and calculate the mean change in the statistic of interest (height or total edge length)
    :param sim_type: "height" or "tbl"
    :param n: simulation sample size (diploids)
    :param n_per_bin: how many simulations to do for each bin (minimum)
    :param bins_start: bin limits
    :param bins_end: bin limits
    :return:
    """
    results_ranked = np.zeros((4, bins_end + 1), dtype=float)
    track_str = ["-"] * (bins_end - bins_start + 1)
    while np.sum(results_ranked[0]) != n_per_bin * (bins_end - bins_start + 1):
        # Note: get correct coalescent time scale with
        # population_size = 0.5, ploidy = 2, n = number of diploids, sequence_length = 1, or
        # population_size = 1, ploidy = 1, n = number of chromosomes, sequence_length = 1
        ts = msprime.sim_ancestry(
            n,
            population_size=0.5,
            ploidy=2,
            recombination_rate=0,
            sequence_length=1,
            random_seed=1 + random.randrange(100000),
        )
        trees = edgespans.compute_trees(ts, 1.0, quiet=True)
        t = ts.first()
        tree = trees.trees[0]
        if sim_type == "height":
            T = math.floor(t.time(t.root))
        else:
            T = math.floor(t.total_branch_length)
        if bins_end >= T >= bins_start:
            if results_ranked[0, T] != n_per_bin:
                results_ranked[0, T] += 1
                if results_ranked[0, T] == n_per_bin:
                    track_str[T - bins_start] = "X"
                    print(" ".join(track_str))
                if sim_type == "height":
                    results_ranked[1, T] += pDnegative(t, tree)
                    results_ranked[2, T] += pDzero(t, tree)
                    results_ranked[3, T] += pDpositive(t, tree)
                else:
                    results_ranked[1, T] += pCnegative(t, tree)
                    results_ranked[2, T] += pCzero(t, tree)
                    results_ranked[3, T] += pCpositive(t, tree)
    print("")
    results_ranked[0, results_ranked[0] == 0] = 1
    results_ranked[1:] = results_ranked[1:] / results_ranked[0]

    return results_ranked
