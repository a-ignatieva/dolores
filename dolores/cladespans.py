import sys
import os
import gc
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import scipy
import math
import tskit
import tszip
import gzip
from . import edgespans


class CladesInfo:
    def __init__(
        self,
        ids,  # index of each corresponding clade within Clades.clades
        num,
        p,
        q,
        log10q,
        sf,
        log10sf,
        q_e,
        log10q_e,
        sf_e,
        log10sf_e,
        D,
        offset,
    ):
        self.num = num
        self.p = p  # P(clade disrupted)
        self.q = q  # Test 1 cdf
        self.log10q = log10q  # Test 1 log(cdf)
        self.sf = sf  # Test 1 sf
        self.log10sf = log10sf  # Test 1 log(sf)
        self.q_event = q_e  # Test 2 cdf
        self.log10q_event = log10q_e  # Test 2 log(cdf)
        self.sf_event = sf_e  # Test 2 sf
        self.log10sf_event = log10sf_e  # Test 2 log(sf)
        self.D = D  # Number of tree breaks in clade span
        self.ids = ids
        self.offset = offset


class Clades_:
    def __init__(self, sequence_length, size):
        self.name = ""
        self.num = 0  # number of non-None clades (added when initially scanning ts)
        self.num_ = 0  # number of (merged) clades with computed p-values
        self.sequence_length = sequence_length
        self.offset = 0.0  # only for argneedle ts

        self.active_clades = {}

        # These are of length = number of (merged) clades with computed p-values
        self.p = []
        self.q = []
        self.log10q = []
        self.sf = []
        self.log10sf = []
        self.q_event = []  # Test 2 cdf
        self.log10q_event = []  # Test 2 log(cdf)
        self.sf_event = []  # Test 2 sf
        self.log10sf_event = []  # Test 2 log(sf)
        self.D = []  # Number of tree breaks in span of clade
        self.ids = []

        # These are indexed by clade ID
        self.id = np.array([-1] * size, dtype=int)
        self.on = np.zeros(size, dtype=int)
        self.nodeid = np.array([-1] * size, dtype=int)
        self.binid = [-1] * size
        self.chunkindex = np.array([-1] * size, dtype=int)
        self.treeindex = np.array([-1] * size, dtype=int)
        self.cladesize = np.array([-1] * size, dtype=int)
        self.tbl = np.array([-1] * size, dtype=float)
        self.start = np.array([-1] * size, dtype=float)
        self.end = np.array([-1] * size, dtype=float)
        self.left_mut = np.array([math.inf] * size, dtype=float)
        self.right_mut = np.array([-1] * size, dtype=float)
        self.num_mutations = np.array([-1] * size, dtype=int)
        self.span = np.array([-1] * size, dtype=float)
        self.mut_span = np.array([-1] * size, dtype=float)
        self.merged = np.zeros(
            size, dtype=int
        )  # how many merged clades represent this one

        # This is a dict of mutations (keys = clade ID)
        self.mutations = defaultdict(
            set
        )  # positions of all mutations on the MRCA edges of this clade

    def add_clade(
        self,
        binid,
        nodeid,
        chunkindex,
        treeindex,
        tbl,
        cladesize,
        start,
    ):
        self.id[self.num] = self.num
        self.on[self.num] = 1
        self.nodeid[self.num] = nodeid
        self.binid[self.num] = binid
        self.chunkindex[self.num] = chunkindex
        self.treeindex[self.num] = treeindex
        self.active_clades[binid] = self.num  # new clade is active
        self.cladesize[self.num] = cladesize
        self.tbl[self.num] = tbl
        self.start[self.num] = start
        self.num_mutations[self.num] = 0
        self.num += 1

        return self.num - 1

    def set_span(self, i, end):
        self.end[i] = end
        self.span[i] = self.end[i] - self.start[i]
        if not (
            (self.left_mut[i] == math.inf or self.right_mut[i] == -1)
            or (self.left_mut[i] == self.right_mut[i])
        ):
            self.mut_span[i] = self.right_mut[i] - self.left_mut[i]

    def add_mutations(self, i, mut):
        if len(mut) > 0:
            self.left_mut[i] = min(self.left_mut[i], min(mut))
            self.right_mut[i] = max(self.right_mut[i], max(mut))
            self.num_mutations[i] += len(mut)
            self.mutations[i].update(mut)

    def print_info(self):
        print(self.num, len(self.active_clades))

    def close(self, end, closed_clades):
        # Set clade ends to sequence length and resize all the arrays
        for key, value in self.active_clades.items():
            self.set_span(value, end)
            closed_clades.append(value)
        self.active_clades = {}
        self.id = np.array(self.id[0 : self.num], dtype=int)
        self.on = np.array(self.on[0 : self.num], dtype=int)
        self.nodeid = np.array(self.nodeid[0 : self.num], dtype=int)
        self.binid = self.binid[0 : self.num]
        self.chunkindex = np.array(self.chunkindex[0 : self.num], dtype=int)
        self.treeindex = np.array(self.treeindex[0 : self.num], dtype=int)
        self.cladesize = np.array(self.cladesize[0 : self.num], dtype=int)
        self.tbl = np.array(self.tbl[0 : self.num], dtype=float)
        self.start = np.array(self.start[0 : self.num], dtype=float)
        self.end = np.array(self.end[0 : self.num], dtype=float)
        self.left_mut = np.array(self.left_mut[0 : self.num], dtype=float)
        self.right_mut = np.array(self.right_mut[0 : self.num], dtype=float)
        self.num_mutations = np.array(self.num_mutations[0 : self.num], dtype=int)
        self.span = np.array(self.span[0 : self.num], dtype=float)
        self.mut_span = np.array(self.mut_span[0 : self.num], dtype=float)
        self.merged = np.array(self.merged[0 : self.num], dtype=int)

    def merge_clades(self, rec_map, cM_limit=0.01):
        # Make a list of clade samples : clade id
        # (This will be in left-to-right order because of the way we insert clades)
        bin_dict = defaultdict(list)
        bin_counts = {}
        warns = 0
        with tqdm(
            total=self.num,
            desc="Merging clades...",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as pbar:
            for i in range(self.num):
                if self.on[i] == 1:
                    if self.binid[i] in bin_counts:
                        # Find position where clade last disappeared
                        cl = max(bin_dict[(self.binid[i], bin_counts[self.binid[i]])])
                        # Get genetic distance
                        Right = self.start[i]
                        Left = self.end[cl]
                        if self.start[i] < rec_map.left[0]:
                            Right = rec_map.left[0]
                        elif self.start[i] > rec_map.right[-1]:
                            Right = rec_map.right[-1]
                        if self.end[cl] < rec_map.left[0]:
                            Left = rec_map.left[0]
                        elif self.end[cl] > rec_map.right[-1]:
                            Left = rec_map.right[-1]
                        if Right != self.start[i] or Left != self.end[cl]:
                            warns += 1
                        d = (
                            rec_map.get_cumulative_mass(Right)
                            - rec_map.get_cumulative_mass(Left)
                        ) * 100
                        if d < cM_limit:
                            bin_dict[(self.binid[i], bin_counts[self.binid[i]])].append(
                                i
                            )
                        else:
                            bin_counts[self.binid[i]] += 1
                            bin_dict[(self.binid[i], bin_counts[self.binid[i]])].append(
                                i
                            )
                    else:
                        bin_dict[(self.binid[i], 0)].append(i)
                        bin_counts[self.binid[i]] = 0
                pbar.update(1)

        print("Warnings:", warns)

        # Merge clades where possible
        num_removed = 0
        for _, clade_list in bin_dict.items():
            if len(clade_list) > 1:
                clade_to_extend = clade_list[0]
                for i, clade in enumerate(clade_list):
                    if i > 0:
                        self.end[clade_to_extend] = self.end[clade]
                        self.num_mutations[clade_to_extend] += self.num_mutations[clade]
                        self.mutations[clade_to_extend].update(self.mutations[clade])
                        if self.num_mutations[clade_to_extend] > 0:
                            self.right_mut[clade_to_extend] = max(
                                self.mutations[clade_to_extend]
                            )
                            self.left_mut[clade_to_extend] = min(
                                self.mutations[clade_to_extend]
                            )
                        self.on[clade] = 0
                        num_removed += 1
                        self.merged[clade_to_extend] += 1
                self.set_span(clade_to_extend, self.end[clade_to_extend])
        print("Merged", num_removed, "clades")

    def reset(self):
        self.num_ = 0  # number of (merged) clades with computed p-values
        self.p = []
        self.q = []
        self.log10q = []
        self.sf = []
        self.log10sf = []
        self.q_event = []
        self.log10q_event = []
        self.sf_event = []
        self.log10sf_event = []
        self.ids = []

    def write_to_file(self, filehandle, clade_indices=None):
        if clade_indices is None or self.num_ != 0:
            clade_indices = [i for i in range(self.num)]
            with gzip.open(filehandle + ".clades.gz", "wt") as file:
                file.write("NAME " + self.name + "\n")
                file.write("NUM_CLADES " + str(self.num) + "\n")
                file.write("NUM_PVALUES " + str(self.num_) + "\n")
                file.write("SEQUENCE_LENGTH " + str(self.sequence_length) + "\n")
                file.write("OFFSET " + str(self.offset) + "\n")
        if self.num_ != len(self.q):
            sys.exit("Error: number of p-values not equal to clades.num_.")
        if self.num_ != 0:
            with gzip.open(filehandle + ".pvalues.gz", "wt") as file:
                file.write("NAME " + self.name + "\n")
                file.write("NUM_CLADES " + str(self.num) + "\n")
                file.write("NUM_PVALUES " + str(self.num_) + "\n")
                file.write("SEQUENCE_LENGTH " + str(self.sequence_length) + "\n")
                file.write("OFFSET " + str(self.offset) + "\n")
                for i in range(self.num_):
                    file.write(
                        str(self.ids[i])
                        + ";"
                        + str(i)
                        + ";"
                        + str(self.p[i])
                        + ";"
                        + str(self.q[i])
                        + ";"
                        + str(self.log10q[i])
                        + ";"
                        + str(self.sf[i])
                        + ";"
                        + str(self.log10sf[i])
                        + ";"
                        + str(self.q_event[i])
                        + ";"
                        + str(self.log10q_event[i])
                        + ";"
                        + str(self.sf_event[i])
                        + ";"
                        + str(self.log10sf_event[i])
                        + ";\n"
                    )
        with gzip.open(filehandle + ".clades.gz", "at") as file:
            for i in clade_indices:
                file.write(
                    str(i)
                    + ";"
                    + str(self.on[i])
                    + ";"
                    + str(self.nodeid[i])
                    + ";"
                    + str(self.binid[i])
                    + ";"
                    + str(self.chunkindex[i])
                    + ";"
                    + str(self.treeindex[i])
                    + ";"
                    + str(self.cladesize[i])
                    + ";"
                    + str(self.tbl[i])
                    + ";"
                    + str(self.start[i])
                    + ";"
                    + str(self.end[i])
                    + ";"
                    + str(self.left_mut[i])
                    + ";"
                    + str(self.right_mut[i])
                    + ";"
                    + str(self.num_mutations[i])
                    + ";"
                    + str(self.span[i])
                    + ";"
                    + str(self.mut_span[i])
                    + ";"
                    + str(self.merged[i])
                    + ";"
                    + " ".join(str(m) for m in self.mutations[i])
                    + ";\n"
                )

    def fix_numbering(self, filehandle):
        if not os.path.exists(filehandle + ".clades.gz"):
            sys.exit("Error: no .clades.gz file found.")
        else:
            with gzip.open(filehandle + "_.clades.gz", "wt") as newfile:
                with gzip.open(filehandle + ".clades.gz", "rt") as oldfile:
                    for l, line in enumerate(oldfile):
                        if l == 1:
                            newfile.write("NUM_CLADES " + str(self.num) + "\n")
                        elif l == 3:
                            newfile.write(
                                "SEQUENCE_LENGTH " + str(self.sequence_length) + "\n"
                            )
                        else:
                            newfile.write(line)
        os.system("mv " + filehandle + "_.clades.gz " + filehandle + ".clades.gz ")


def read_from_file(filehandle):
    filename = filehandle + ".clades.gz"
    if not os.path.exists(filename):
        sys.exit("Error: no .clades.gz file found.")
    else:
        info = [""] * 5
        with gzip.open(
            filename,
            "rt",
        ) as file:
            for l, line in enumerate(file):
                if l <= 4:
                    line = line.strip().split(" ")
                    print(line)
                    if len(line) > 1:
                        info[l] = line[1]
                else:
                    if l == 5:
                        clades = Clades_(float(info[3]), int(info[1]))
                        clades.name = info[0]
                        clades.num = int(info[1])
                        clades.num_ = int(info[2])
                        clades.offset = float(info[4])
                    line = line.strip().split(";")
                    i = int(line[0])
                    clades.id[i] = i
                    clades.on[i] = int(line[1])
                    clades.nodeid[i] = int(line[2])
                    clades.binid[i] = line[3]
                    clades.chunkindex[i] = int(line[4])
                    clades.treeindex[i] = int(line[5])
                    clades.cladesize[i] = int(line[6])
                    clades.tbl[i] = float(line[7])
                    clades.start[i] = float(line[8])
                    clades.end[i] = float(line[9])
                    clades.left_mut[i] = float(line[10])
                    clades.right_mut[i] = float(line[11])
                    clades.num_mutations[i] = int(line[12])
                    clades.span[i] = float(line[13])
                    clades.mut_span[i] = float(line[14])
                    clades.merged[i] = int(line[15])
                    if line[16] != "":
                        clades.mutations[i] = set(
                            {float(m) for m in line[16].strip().split(" ")}
                        )

                    if (
                        clades.start[i] == -1
                        or clades.end[i] == -1
                        or clades.span[i] <= 0
                        or (clades.num_mutations[i] >= 2 and clades.mut_span[i] == -1)
                    ):
                        sys.exit(
                            "Error: clade start, end or span is not recorded properly."
                        )
                    if clades.mut_span[i] > clades.span[i]:
                        sys.exit("Error: mut_span cannot be greater than span.")
                    if clades.start[i] > clades.end[i]:
                        sys.exit("Error: clade start cannot be greater than clade end.")
                    if (
                        clades.num_mutations[i] > 0
                        and clades.right_mut[i] < clades.left_mut[i]
                    ):
                        sys.exit("Error: left_mut cannot be greater than right_mut.")
    filename = filehandle + ".pvalues.gz"
    if not os.path.exists(filename):
        print("Cannot read in p-values (file not found).")
    else:
        with gzip.open(
            filename,
            "rt",
        ) as file:
            for l, line in enumerate(file):
                if l >= 5:
                    line = line.strip().split(";")
                    # file.write("clade_num;clade_id;clade_prob;clade_cdf;clade_log10cdf;clade_p;clade_log10p")
                    clades.ids.append(int(line[0]))
                    clades.q.append(float(line[3]))
                    clades.p.append(float(line[2]))
                    clades.log10q.append(float(line[4]))
                    clades.sf.append(float(line[5]))
                    clades.log10sf.append(float(line[6]))
                    clades.q_event.append(float(line[7]))
                    clades.log10q_event.append(float(line[8]))
                    clades.sf_event.append(float(line[9]))
                    clades.log10sf_event.append(float(line[10]))
    if clades.num_ != len(clades.q):
        sys.exit("Error: wrong number of lines in .pvalues file or wrong .num_.")

    print("Read in", clades.num, "clades")

    return clades


def add_info(
    clades,
    cladesInfo,
):
    if cladesInfo.num != len(cladesInfo.q):
        sys.exit("Error: mismatching number of entries in cladesInfo.")
    clades.num_ += cladesInfo.num
    clades.q.extend(cladesInfo.q)
    clades.p.extend(cladesInfo.p)
    clades.log10q.extend(cladesInfo.log10q)
    clades.sf.extend(cladesInfo.sf)
    clades.log10sf.extend(cladesInfo.log10sf)
    clades.q_event.extend(cladesInfo.q_event)
    clades.log10q_event.extend(cladesInfo.log10q_event)
    clades.sf_event.extend(cladesInfo.sf_event)
    clades.log10sf_event.extend(cladesInfo.log10sf_event)
    clades.ids.extend(cladesInfo.ids)
    if clades.offset != 0 and clades.offset != cladesInfo.offset:
        sys.exit("Error: offset of results chunk does not match that already recorded.")
    clades.offset = cladesInfo.offset


def calculate_q(
    results_chunk,
    rec_map,
    ts_chunk,
    trees_chunk,
    breakpoints,
    argn_trees=False,
    argn_ts=None,
    use_muts=False,
    muts_per_kb=0,
    destroy_trees=False,  # Delete the tree info as we go to save memory
):
    """
    Compute p-values for Test 1 and Test 2
    """

    if argn_trees and argn_ts is None:
        sys.exit("Need ARG-Needle tree sequence")

    size = len(results_chunk)
    P = np.zeros(size, dtype=float)
    Q = np.zeros(size, dtype=float)
    log10Q = np.zeros(size, dtype=float)
    SF = np.zeros(size, dtype=float)
    log10SF = np.zeros(size, dtype=float)
    Q_e = np.zeros(size, dtype=float)
    log10Q_e = np.zeros(size, dtype=float)
    SF_e = np.zeros(size, dtype=float)
    log10SF_e = np.zeros(size, dtype=float)
    D = np.zeros(size, dtype=float)
    ids = []
    offset = 0
    prevtree = -1
    if argn_trees:
        offset = int(argn_ts.metadata["offset"]) - 1
    i = 0
    t = tskit.Tree(ts_chunk)

    with tqdm(total=size, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
        for ch in results_chunk:
            clade_id = ch[0]
            clade_nodeid = ch[1]
            clade_treeindex = ch[2]
            clade_tbl = ch[3]
            clade_mut_span = ch[4]
            clade_left_mut = ch[5]
            clade_right_mut = ch[6]
            clade_dur = ch[7]
            clade_start = ch[8]
            clade_end = ch[9]
            clade_num_mutations = ch[10]

            if use_muts:
                d = clade_mut_span
                start = clade_left_mut
                end = clade_right_mut
                if clade_num_mutations <= 2:
                    d = -1
            else:
                d = clade_dur
                start = clade_start
                end = clade_end
            if d != -1 and start != -1 and end != -1:
                if 1000 * clade_num_mutations / d >= muts_per_kb:
                    if destroy_trees:
                        if prevtree != -1 and prevtree != clade_treeindex:
                            del tree
                            trees_chunk.trees[prevtree] = None
                            gc.collect()
                        prevtree = clade_treeindex
                    t.seek_index(clade_treeindex)
                    tree = trees_chunk.trees[clade_treeindex]
                    P[i] = prob_g_disrupted(clade_nodeid, t, tree)
                    Left = max(start + offset, rec_map.left[0])
                    Right = min(end + offset, rec_map.right[-1])
                    if Right < Left:
                        # print("Warning: Right < Left for clade", clade_id)
                        R = 0
                    else:
                        R = rec_map.get_cumulative_mass(
                            Right
                        ) - rec_map.get_cumulative_mass(Left)
                    log10Q[i] = scipy.stats.expon.logcdf(
                        P[i] * clade_tbl * R, loc=0, scale=1
                    ) / np.log(10)
                    Q[i] = scipy.stats.expon.cdf(P[i] * clade_tbl * R, loc=0, scale=1)
                    log10SF[i] = scipy.stats.expon.logsf(
                        P[i] * clade_tbl * R, loc=0, scale=1
                    ) / np.log(10)
                    SF[i] = scipy.stats.expon.sf(P[i] * clade_tbl * R, loc=0, scale=1)
                    l = np.where(breakpoints <= start)[0][-1]
                    r = np.where(breakpoints <= end)[0][-1]
                    dd = r - l + 1
                    D[i] = int(dd)
                    Q_e[i] = scipy.stats.geom.cdf(dd, P[i])
                    log10Q_e[i] = scipy.stats.geom.logcdf(dd, P[i])
                    SF_e[i] = scipy.stats.geom.sf(dd - 1, P[i])
                    log10SF_e[i] = scipy.stats.geom.logsf(dd - 1, P[i]) / np.log(10)
                    ids.append(clade_id)
                    i += 1

            pbar.update(1)

    cladesInfo = CladesInfo(
        ids=ids,  # index of each corresponding clade within Clades.clades
        num=i,
        p=P[0:i],
        q=Q[0:i],
        log10q=log10Q[0:i],
        sf=SF[0:i],
        log10sf=log10SF[0:i],
        q_e=Q_e[0:i],
        log10q_e=log10Q_e[0:i],
        sf_e=SF_e[0:i],
        log10sf_e=log10SF_e[0:i],
        D=D,
        offset=offset,
    )

    return cladesInfo


def find_treeindices(breakpoints, pos1, pos2):
    t1 = np.where(breakpoints <= pos1)[0][-1]
    t2 = np.where(breakpoints <= pos2)[0][-1]
    return t1, t2


def ltt_g(g, t, tree, verbose=False):
    """
    Lineages through time that do and do not belong to clade under edge G
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param verbose: print info
    :return:
    """
    num_G_lineages = np.zeros(len(tree.num_lineages), dtype=int)
    num_Gg_lineages = np.zeros(len(tree.num_lineages), dtype=int)
    count = 0
    gup = None
    Gup = None
    for i, n in enumerate(t.nodes(order="timedesc")):
        if verbose:
            print("i =", i, "node = ", n)
        if t.time(n) == 0:
            if verbose:
                print("reached samples: recording", count, "and stopping")
            num_G_lineages[i] = count
            if not Gup:
                Gup = i + 1
            break
        elif t.is_descendant(n, g):
            num_G_lineages[i] = count
            if count == 0:
                count = 1
                Gup = i + 1
            if verbose:
                print("descendant, recording", count)
            count += 1
        elif n == t.parent(g):
            gup = i + 1
        else:
            if verbose:
                print("not a descenant, recording", count)
            num_G_lineages[i] = count
    for i in range(len(num_G_lineages)):
        if gup <= i < Gup:
            num_Gg_lineages[i] = 1
        else:
            num_Gg_lineages[i] = num_G_lineages[i]
    if verbose:
        print(
            "num_G_lineages:",
            num_G_lineages,
            "num_Gg_lineages:",
            num_Gg_lineages,
            "gup:",
            gup,
            "Gup:",
            Gup,
        )
    if len(num_G_lineages) != t.num_samples():
        sys.exit("Error: len(num_G_lineages) wrong in tree " + str(t.index))
    return num_G_lineages, num_Gg_lineages, Gup, gup


def Ls_g(g, t, tree):
    """
    Total edge length of clade below g in generations and on coalescent scale
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    L1 = sum(t.branch_length(u) for u in t.nodes(root=g) if u != g)
    L2 = sum(
        tree.time_dict[t.time(t.parent(u))] - tree.time_dict[t.time(u)]
        for u in t.nodes(root=g)
        if u != g
    )
    return L1, L2


def G_term2(k, lineages, up, tree, verbose=False):
    # Tk = tree.tim_lineages[k - 2]
    # Tk1 = tree.tim_lineages[k - 1]
    p = 0
    for j in range(1 + up, k + 1):
        nj1 = lineages[j - 1]
        q = tree.lookup_Qkj(k, j)
        p += nj1 * q
        if verbose:
            print(
                "Term 2: summing", j, k, ", number of lineages = ", nj1, "up = ", up, p
            )
    p = p * 1 / k
    return p


def prob_CnotinGg_RinG(
    g, t, tree, num_G_lineages, num_Gg_lineages, Gup, gup, verbose=False
):
    """
    P(C not in G u g | R in G)
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param num_G_lineages: lineages through time for edges in clade G
    :param num_Gg_lineages: lineages through time for edges in clade G including edge above g
    :param Gup: number of lineages at time of clade MRCA
    :param gup: number of lineages at upper end of g edge
    :param verbose: print info
    :return:
    """
    if t.is_sample(g):
        return 1.0
    else:
        down = t.num_samples()
        P = 0
        L1, L2 = Ls_g(g, t, tree)
        for k in range(Gup + 1, down + 1):
            nGTk1 = num_G_lineages[k - 1]
            nGgTk1 = num_Gg_lineages[k - 1]
            k = tree.num_lineages[k - 1]
            Tk = tree.tim_lineages[k - 2]
            Tk1 = tree.tim_lineages[k - 1]
            Tkgen = tree.time_dict_rev[tree.tim_lineages[k - 2]]
            Tk1gen = tree.time_dict_rev[tree.tim_lineages[k - 1]]
            q1 = edgespans.Q1(k, tree)
            term2 = G_term2(k, num_Gg_lineages, gup, tree, verbose)

            if Tk != Tk1:
                termi = 0
                for N, t1, t2 in zip(
                    tree.time_grid[k][1],
                    tree.time_grid[k][0][:-1],
                    tree.time_grid[k][0][1:],
                ):
                    termi += N * (np.exp(k * t2) - np.exp(k * t1))
                P += nGTk1 * (
                    nGgTk1 * q1 * (Tkgen - Tk1gen) / (Tk - Tk1) + term2 * termi
                )
        P = P / L1
        P = 1 - P

        return P


def prob_CinG_RnotinGg(g, t, tree, num_G_lineages, num_Gg_lineages, Gup, verbose=False):
    """
    P(C in G | R not in G u g)
    :param g: node id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param num_G_lineages: lineages through time for edges in clade G
    :param num_Gg_lineages: lineages through time for edges in clade G including edge above g
    :param Gup: number of lineages at time of clade MRCA
    :param verbose: print info
    :return:
    """
    L1, L2 = Ls_g(g, t, tree)
    if tree.tbl <= L2:
        return 1.0
    else:
        tg = t.time(t.parent(g)) - t.time(g)
        down = t.num_samples()
        P = 0
        for k in range(Gup + 1, down + 1):
            Tk = tree.tim_lineages[k - 2]
            Tk1 = tree.tim_lineages[k - 1]
            k = tree.num_lineages[k - 1]
            Tkgen = tree.time_dict_rev[tree.tim_lineages[k - 2]]
            Tk1gen = tree.time_dict_rev[tree.tim_lineages[k - 1]]
            # print("k = ", k)
            nGTk1 = num_G_lineages[k - 1]
            nGgTk1 = num_Gg_lineages[k - 1]

            if Tk != Tk1:
                term2 = G_term2(k, num_G_lineages, Gup, tree, verbose)
                q1 = edgespans.Q1(k, tree)
                termi = 0
                for N, t1, t2 in zip(
                    tree.time_grid[k][1],
                    tree.time_grid[k][0][:-1],
                    tree.time_grid[k][0][1:],
                ):
                    termi += N * (np.exp(k * t2) - np.exp(k * t1))
                P += (k - nGgTk1) * (
                    nGTk1 * q1 * (Tkgen - Tk1gen) / (Tk - Tk1) + term2 * termi
                )
        P = P / (t.total_branch_length - L1 - tg)

        return P


def prob_RinG(g, t, tree):
    """
    Probability of recombination event in clade G
    :param g: edge id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    L1, L2 = Ls_g(g, t, tree)
    return L1 / t.total_branch_length


def prob_RnotinGg(g, t, tree):
    """
    Probability of recombination event not in clade G or on edge above g
    :param g: edge id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    tg = t.time(t.parent(g)) - t.time(g)
    L1, L2 = Ls_g(g, t, tree)
    return (t.total_branch_length - L1 - tg) / t.total_branch_length


def prob_Ring(g, t, tree):
    """
    Probability of recombination event on edge above g
    :param g: edge id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    tg = t.time(t.parent(g)) - t.time(g)
    return tg / t.total_branch_length


def prob_g_disrupted(g, t, tree, test=False, verbose=False):
    """
    Probability clade under g is disrupted by the recombination event
    :param g: edge id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :param test: whether to check that all calculated probabilities sum to 1
    :param verbose: print info
    :return:
    """
    num_G_lineages, num_Gg_lineages, Gup, gup = ltt_g(g, t, tree)
    P1 = prob_RinG(g, t, tree)
    P2 = prob_RnotinGg(g, t, tree)
    P3 = prob_CnotinGg_RinG(
        g, t, tree, num_G_lineages, num_Gg_lineages, Gup, gup, verbose
    )
    P4 = prob_CinG_RnotinGg(g, t, tree, num_G_lineages, num_Gg_lineages, Gup, verbose)

    if verbose:
        print(P1, P2, P3, P4)
    if test:
        if abs(prob_g_test(g, t, tree) - 1) > 1e-5:
            sys.exit("Error: test not passed with threshold 1e-5")
        else:
            print("Test passed")
    return P1 * P3 + P2 * P4


def prob_g_test(g, t, tree):
    """
    Testing that all the probabilities we calculate sum to 1
    :param g: edge id of clade MRCA
    :param t: tree in tskit format
    :param tree: tree object
    :return:
    """
    num_G_lineages, num_Gg_lineages, Gup, gup = ltt_g(g, t, tree)
    P0 = prob_Ring(g, t, tree)
    P1 = prob_RinG(g, t, tree)
    P2 = prob_RnotinGg(g, t, tree)
    P3 = prob_CnotinGg_RinG(g, t, tree, num_G_lineages, num_Gg_lineages, Gup, gup)
    P4 = prob_CinG_RnotinGg(g, t, tree, num_G_lineages, num_Gg_lineages, Gup)
    return P1 * P3 + P2 * P4 + P1 * (1 - P3) + P2 * (1 - P4) + P0


def get_bitset(n, t, bitset, samp_to_ind):
    """
    Get "genotype" id of a node
    This is not unique though
    """
    e = [0] * int(len(samp_to_ind) / 2)
    if t.is_sample(n):
        e[samp_to_ind[n]] = 1
    else:
        for ch in t.children(n):
            for i, s in enumerate(bitset[ch]):
                if s != 0:
                    e[i] = s + e[i]
    return e


def find_clade(b, tree, samp_to_ind):
    bitset = {}
    c2 = -1
    for g in tree.nodes(order="timeasc"):
        g_id = get_bitset(g, tree, bitset, samp_to_ind)
        bitset[g] = g_id
        g_id = "".join(str(s) for s in g_id)
        if g_id == b:
            c2 = g
            break
    return c2


def clade_span(
    ts_list,
    num_trees,
    num_samples,
    samp_to_ind=None,
    write_to_file=None,
    write_to_file_freq=None,
):
    """
    Compute the span of clades in a given tree sequence.
    This is defined by the samples subtending a clade staying the same.
    This WILL NOT calculate the clade disruption probabilities (that way those can be done in parallel,
    and only for the relevant clades)
    :param ts_list: list of tskit tree sequence file handles with extensions (will be loaded as we go)
    :param num_trees: total number of trees in full ts (excluding the empty trees)
    :param num_samples: number of samples in each ts
    :param samp_to_ind: dictionary of {sample node ID : individual ID}
    :param write_to_file: filename for outputting clades periodically
    :param write_to_file_freq: how often to output clades to file
    :return:
    """

    clades = Clades_(None, size=int(num_trees * (num_samples - 2) / 2))
    closed_clades = []
    duplicate_clades = []
    tree_counter = -1
    left = 0.0
    if write_to_file is not None:
        with gzip.open(write_to_file + ".clades.gz", "wt") as file:
            file.write("NAME " + clades.name + "\n")
            file.write("NUM_CLADES 0\n")
            file.write("NUM_PVALUES 0\n")
            file.write("SEQUENCE_LENGTH None\n")
            file.write("OFFSET 0.0\n")

    with tqdm(
        total=num_trees,
        desc="Computing clade spans",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ) as pbar:
        for i, ts_handle in enumerate(ts_list):
            if os.path.exists(ts_handle + ".tsz"):
                ts = tszip.decompress(ts_handle + ".tsz")
            else:
                ts = tskit.load(ts_handle)
            if i == len(ts_list) - 1:
                clades.sequence_length = ts.sequence_length

            for t in ts.trees():
                if (
                    t.num_roots == 1
                ):  # Sometimes first tree in ts out of stdpopsim is empty so skip it
                    tree_counter += 1
                    left = max(
                        left, t.interval[0]
                    )  # This is for having multiple Relate ts parts
                    bitset = {}  # record node IDs as an array
                    bitset_ = set()  # record node IDs as a string
                    prevclades = list(
                        clades.active_clades.values()
                    )  # clades in previous tree
                    tree_muts = defaultdict(set)

                    # Dictionary of {node_tskit_id : set(mutation_positions)}
                    for mut in t.mutations():
                        tree_muts[mut.node].add(ts.site(mut.site).position)

                    for g in t.nodes(order="timeasc"):
                        if g != t.root:
                            if t.is_sample(g):
                                # Just record the bitset
                                if samp_to_ind is None:
                                    g_id = 1 << g
                                else:
                                    g_id = get_bitset(g, t, bitset, samp_to_ind)
                                    bitset_.add("".join(str(s) for s in g_id))
                                bitset[g] = g_id
                            elif samp_to_ind is not None and t.num_samples(g) == 2:
                                # We will ignore size-2 clades because these are not unlikely enough
                                # to have identical genotype IDs. So just record the ID of this node.
                                g_id = get_bitset(g, t, bitset, samp_to_ind)
                                bitset[g] = g_id
                                bitset_.add("".join(str(s) for s in g_id))
                            else:
                                m = tree_muts[
                                    g
                                ]  # Set of mutation positions above this node in this tree

                                if samp_to_ind is None:
                                    g_id = 0
                                    for ch in t.children(g):
                                        g_id = g_id | bitset[ch]
                                    cladesize = g_id.bit_count()
                                    bitset[g] = g_id
                                else:
                                    g_id = get_bitset(g, t, bitset, samp_to_ind)
                                    cladesize = np.sum(g_id)
                                    bitset[g] = g_id
                                    g_id = "".join(str(s) for s in g_id)
                                    if g_id in bitset_:
                                        # This is a duplicate based on genotype ID
                                        duplicate_clades.append(
                                            [
                                                i,
                                                t.index,
                                                tree_counter,
                                                g,
                                                cladesize,
                                                [s for s in t.samples(g)],
                                                [samp_to_ind[s] for s in t.samples(g)],
                                            ]
                                        )
                                    bitset_.add(g_id)

                                if t.num_samples(g) != cladesize:
                                    sys.exit(
                                        "Error: clade size per genotype ID does not match tree."
                                    )

                                if g_id in clades.active_clades:
                                    # The clade was there in the previous tree, so record mutations and
                                    # remove from prevclades (so prevclades will become the list of
                                    # clades that have disappeared).
                                    p = clades.active_clades[g_id]
                                    clades.add_mutations(p, m)
                                    if p in prevclades:
                                        prevclades.remove(p)
                                    # Add number of mutations above g to clade mutation count
                                else:
                                    # This is a new clade
                                    # (this will be recorded as active when added)
                                    p = clades.add_clade(
                                        binid=g_id,  # this is number or string
                                        nodeid=g,
                                        chunkindex=i,
                                        treeindex=t.index,
                                        tbl=t.total_branch_length,
                                        cladesize=cladesize,
                                        start=left,
                                    )
                                    clades.add_mutations(p, m)
                    for p in prevclades:
                        # These clades have disappeared
                        clades.set_span(p, left)
                        closed_clades.append(p)

                    clades.active_clades = {
                        key: val
                        for key, val in clades.active_clades.items()
                        if val not in prevclades
                    }

                    if i == len(ts_list) - 1 and t.index == ts.num_trees - 1:
                        clades.close(t.interval[1], closed_clades)
                        if write_to_file is not None:
                            clades.write_to_file(write_to_file, closed_clades)
                            clades.fix_numbering(write_to_file)
                        pbar.update(1)
                        break

                    if write_to_file is not None:
                        if len(closed_clades) >= write_to_file_freq:
                            clades.write_to_file(write_to_file, closed_clades)
                            for d in closed_clades:
                                del clades.mutations[d]
                                clades.binid[d] = None
                            closed_clades = []

                    pbar.update(1)

    print("Done, traversed", tree_counter + 1, "trees, out of", num_trees)
    clades.print_info()

    print("Clades that are duplicates based on genotype ID:")
    print(
        "chunk;chunk_tree_index;tree_index;node_id;cladesize;sample_ids;individual_ids"
    )
    for d in duplicate_clades:
        print(*d, sep=";")

    if write_to_file is not None:
        clades = None

    return clades, duplicate_clades
