import stdpopsim
import tskit
import tsinfer

# from asmc.preparedecoding import *
# import tsdate
import math
import numpy as np
import sys
import os
import random


def simulate_data(
    contig, model, samples, sim_model="hudson", rs=-1, record_full_arg=False
):
    if rs == -1:
        rs = 1 + random.randrange(1000000)
        print("seed:", rs)
    engine = stdpopsim.get_engine("msprime")

    print("simulating...", end="", flush=True)
    ts = engine.simulate(
        model,
        contig,
        samples,
        msprime_model=sim_model,
        random_seed=rs,
        record_full_arg=record_full_arg,
    )
    print("done", flush=True)

    print("trees: ", ts.num_trees)
    print("length: ", ts.sequence_length)
    print("mutations: ", ts.num_mutations)
    print("sample sequences: ", ts.num_samples)
    print("simulation model:", sim_model, flush=True)

    return ts, rs


def clean_relate():
    """
    Clears old Relate files
    :return:
    """
    os.system("rm -r relate*")
    os.system("rm relate_sim*")
    os.system("rm relate.*")
    os.system("rm ancestral_genome.fa")
    os.system("rm samples.vcf")
    os.system("rm sample_ages.txt")
    os.system("rm out.txt")


def run_relate(
    ts,
    rec_map_file,
    mutation_rate,
    Ne,
    relate_loc,
    mask=None,
    consistency=False,
    postprocess=False,
    randomise=False,
    memory=5,
    threads=1,
    quiet=False,
):
    """
    Runs Relate on the input tree sequence
    :param threads: how many threads for Relate
    :param memory: RAM per thread
    :param ts:
    :param rec_map_file:
    :param mutation_rate:
    :param Ne: effective population size of HAPLOIDS
    :param relate_loc:
    :param consistency:
    :param postprocess:
    :param randomise:
    :param quiet:
    :return:
    """
    clean_relate()

    print("Writing files...", end="", flush=True)
    with open("samples.vcf", "w") as vcf_file:
        ts.write_vcf(vcf_file)

    if mask is not None:
        mask = " --mask " + mask

    # Hacky: ancestral genome is A except where we have mutations so know the ancestral state from the ts.
    ancestral_genome = ["A"] * int(ts.sequence_length)
    with open("ancestral_genome.fa", "w") as file:
        file.write(">ancestral_sequence\n")
        for s in ts.sites():
            ancestral_genome[int(s.position) - 1] = s.ancestral_state
        file.write("".join(ancestral_genome))
        file.write("\n")

    sample_ages = np.zeros(ts.num_samples)
    # for k, pop in enumerate(ts.populations()):
    #     if len(ts.samples(k)) > 0 and pop.metadata["sampling_time"] != 0:
    #         for i in ts.samples(k):
    #             sample_ages[i] = pop.metadata["sampling_time"]
    np.savetxt("sample_ages.txt", sample_ages, delimiter="\n")
    print("done", flush=True)

    S = "relate"
    Si = S + "_input"

    if not quiet:
        os.system(
            relate_loc
            + "/bin/RelateFileFormats --mode ConvertFromVcf --haps "
            + S
            + ".haps --sample "
            + S
            + ".sample -i samples;"
        )
        os.system(
            relate_loc
            + "/scripts/PrepareInputFiles/PrepareInputFiles.sh --haps "
            + S
            + ".haps --sample "
            + S
            + ".sample --ancestor ancestral_genome.fa -o "
            + Si
            + mask
            + ";"
        )
        if consistency:
            cons = "--consistency "
            So = S + "_con"
        else:
            cons = ""
            So = S
        if threads == 1:
            os.system(
                relate_loc
                + "/bin/Relate --mode All --sample_ages sample_ages.txt "
                + cons
                + "--memory "
                + str(memory)
                + " -m "
                + str(mutation_rate)
                + " -N "
                + str(Ne)
                + " --haps "
                + Si
                + ".haps.gz --sample "
                + Si
                + ".sample.gz --map "
                + str(rec_map_file)
                + " --seed 1 -o "
                + S
                + ";"
            )
        else:
            os.system(
                relate_loc
                + "/scripts/RelateParallel/RelateParallel.sh --sample_ages sample_ages.txt "
                + cons
                + "--memory "
                + str(memory)
                + " -m "
                + str(mutation_rate)
                + " -N "
                + str(Ne)
                + " --haps "
                + Si
                + ".haps.gz --sample "
                + Si
                + ".sample.gz --map "
                + str(rec_map_file)
                + " --seed 1 -o "
                + S
                + " --threads "
                + str(threads)
                + " ;"
            )
        if postprocess:
            Sop = So + "_pp"
            if randomise:
                ran = " --randomise "
                Sop += "_ran"
            else:
                ran = ""
            os.system(
                relate_loc
                + "/bin/Relate --mode PostProcess "
                + ran
                + " --input "
                + S
                + " --haps "
                + Si
                + ".haps.gz --sample "
                + Si
                + ".sample.gz -o "
                + Sop
                + ";"
            )


def get_vcf_haps_sample(
    ts,
    sample_path,
    hap_path,
    chrom,
):
    """
    Get .haps and .sample file for input ts
    :param ts: input tree sequence in tskit format
    :param relate_loc: location of Relate
    :param filehandle: filehandle to write files, no extension
    :return:
    """
    with open(sample_path, "w+") as samplefile:
        samplefile.write("ID_1 ID_2 missing\n0 0 0\n")
        for i in range(1, ts.num_individuals + 1):
            samplefile.write(f"s{i} s{i} 0\n")

    index = 0
    with open(hap_path, "wb+") as hapfile:
        for i, v in enumerate(ts.variants()):
            snp_id = f"{i}"
            # we add the start position argument here
            bp = v.position + ts.breakpoints(as_array=True)[0]
            genotypes = v.genotypes.tolist()
            geno_str = " ".join([str(g) for g in genotypes])
            hapfile.write(f"{chrom} {snp_id} {bp:.0f} G A {geno_str}\n".encode())
            index += 1


# Function to get recombination map from rec_map stdpopsim object


def get_map(ts, rec_map, chromosome_code="chr0", filehandle="simulated_data"):
    """
    Get variant information in plink .map format for ARG-Needle
    :param ts: input tree sequence in tskit format
    :param rec_map: msprime recombination map object
    :param chromosome_code: chromosome name
    :param filehandle: filehandle to write .map file (no extension)
    :return:
    """
    chrom = int(list(filter(str.isdigit, chromosome_code))[0])
    # offset = rec_map.left[1]
    with open(filehandle + ".map", "w") as file:
        for s in ts.sites():
            P = s.position
            R = rec_map.get_cumulative_mass(P) * 100
            file.write(
                str(chrom)
                + "\t"
                + str(s.id)
                + "\t"
                + str(R)
                + "\t"
                + str(int(P))
                + "\n"
            )


# Function to fix recombination map in such a way that genetic distance entries are strictly increasing, as required by argneedle


def fix_argneedle_map(map_filename):
    """
    Call recombination map "argneedle.map" and ensure that genomic distance
    entries are strictly increasing -- necessary for argneedle to run
    """
    epsilon = 1e-4  # This is the smallest possible value to make this work
    # Load in original recombination map
    recombination_map = np.loadtxt(map_filename, delimiter="\t")
    lastR, addonR = 0, 0
    for s in range(np.shape(recombination_map)[0]):
        # If diff small, then add in epsilon
        if recombination_map[s, 2] - lastR < epsilon:
            addonR += (
                epsilon  # Keep track of epsilon as genetic distance is "cumulative"
            )
            recombination_map[s, 2] += addonR
        lastR = recombination_map[s, 2]
    rec_file_format = "%d", "%d", "%1.10f", "%d"
    np.savetxt(
        map_filename, recombination_map, delimiter="\t", fmt=rec_file_format
    )  # Save to file in required format


# Function to invoke argneedle reconstruction
def run_argneedle(
    ts,
    recombination_map,
    mutation_rate,
    chromosome,
    decoding_demo_filename,
    time_discretisation,
    N_samples,
    model,
    resource_loc,
    simulation_loc,
    filehandle="argneedle",
):
    """
    Runs argneedle
    First creates a .decodingQuantities.gz file for use within ASMC,
    then uses the .haps, .sample, and .map files provided to run argneedle,
    and finally converts resulting .argn file to tskit format
    :param ts: tree-sequence we are trying to reconstruct,
    :param haps_filename: filename of haploids extracted from simulated data,
    :param map_filename: filename of recombination map for simulated data,
    :param decoding_demo_filename: filename of demography file used for ASMC decoding,
    :param model: which model we are using (e.g. smc_prime)
    :param time_disrectisation: time discretisation bins (quantiles) used within ASMC decoding,
    :param N_samples: number of samples within ARG,
    :param mut_rate: mutation rate,
    :param filehandle: filehandle identifier for argneedle output
    """
    get_vcf_haps_sample(
        ts, "argneedle.sample", "argneedle.haps", chromosome
    )  # Load in .haps and .sample files
    argneedle_map_filename = str(filehandle) + ".map"  # Set map filehandle
    get_map(
        ts, recombination_map, chromosome, filehandle=filehandle
    )  # Get the recombination map
    # Ensure genetic positions are strictly increasing
    fix_argneedle_map(argneedle_map_filename)
    # Call ASMC to prepare decoding files to be used within argneedle
    dq = prepare_decoding(
        demography=decoding_demo_filename,
        discretization=[time_discretisation],
        frequencies=resource_loc + "UKBB.frq",
        samples=N_samples,
        mutation_rate=mutation_rate,
    )
    out_root = str("const10k.N" + str(N_samples))  # Set decoding files path
    # Print out relevant decoding quantities to file
    dq.save_decoding_quantities(out_root)
    dq.save_csfs(out_root)
    dq.save_intervals(out_root)
    dq.save_discretization(out_root)
    asmc_decoding_filename = (
        out_root + ".decodingQuantities.gz"
    )  # Set decoding filehandle
    output_filename = (
        simulation_loc + filehandle + "_" + str(model) + "_" + str(N_samples)
    )  # Set output filehandle
    # Invoke argneedle with the relevant .haps, .sample and .map files, in sequence mode, without arg normalisation, setting sequence_hash_cm to 0.3 and using the correct ASMC decoding files
    os.system(
        "arg_needle --hap_gz argneedle.haps --map argneedle.map --out "
        + str(output_filename)
        + " --mode sequence --normalize 0 --sequence_hash_cm 0.3 --asmc_decoding_file "
        + str(asmc_decoding_filename)
    )
    # Convert .argn output from above into tskit .trees format
    os.system(
        "arg2tskit --arg_path "
        + str(output_filename)
        + ".argn --ts_path "
        + str(output_filename)
        + ".trees"
    )
    # Procedure to squash any continguous edges which can be merged (i.e. same children and parent, contiguous genomic intervals)
    ts_unsquashed = tskit.load(output_filename + ".trees")
    tables = ts_unsquashed.tables
    tables.edges.squash()
    tables.sort()
    ts_squashed = tables.tree_sequence()
    # Save squashed tree to file
    ts_squashed.dump(output_filename + "_squashed.trees")
    return None


def read_mut_file(filename, sequence_length):
    """
    Read site positions from .mut file
    :param filename: filehandle, no extension
    :param sequence_length: sequence length
    :return: list of positions corresponding to each mutation site
    """
    with open(filename + ".mut", "r") as file:
        N = sum(1 for _ in file) - 1

    sites = np.zeros(N + 2)
    edges = {}
    with open(filename + ".mut", "r") as file:
        next(file)
        for i, line in enumerate(file.readlines()):
            ch = line.split(";")
            sites[i + 1] = int(ch[1])
            ch5 = ch[5].split(" ")
            if not ch5[0]:
                # Mutation is not mapped to any edge, skip
                continue
            else:
                edge_indices = [int(b) for b in ch5]
                if len(edge_indices) == 1:
                    edges[int(ch[1])] = (int(ch[4]), edge_indices[0])

    sites[N + 1] = sequence_length
    return sites, edges


def read_anc_file(
    filename, sequence_length, supported_only=True, pos_range=(-math.inf, math.inf)
):
    """
    Read edge durations from .anc file
    :param supported_only:
    :param pos_range:
    :param filename: filehandle, no extension
    :param sequence_length: sequence length
    :return: edges_relate is a dict of edge IDs {unique edge : id in tskit tree}
    edge_durations_relate is a 2D list of edge left and right endpoints
    edge_lengths_relate is a list of edge time-lengths
    """
    sites, _ = read_mut_file(filename, sequence_length)
    k = 0
    kk = 0
    with open(filename + ".anc", "r") as file:
        for i, line in enumerate(file.readlines()):
            if i == 0:
                ch = line.split()
                num_samples = int(ch[1])
            elif i == 1:
                ch = line.split()
                num_trees = int(ch[1])
                print("Trees:", num_trees)
                print("Samples:", num_samples)
                edges_relate = {}
                edge_durations_relate = np.zeros(
                    (num_trees, 2 * num_samples - 2, 2), dtype=float
                )
                edge_lengths_relate = np.zeros(
                    (num_trees, 2 * num_samples - 2), dtype=float
                )
            else:
                tree_start = int(line.split(":")[0])
                line = line.replace(")", "+")
                line = line[:-3]
                line = line.strip()
                for j, ch in enumerate(line.split("+")):
                    ch = ch.replace(":", " ")
                    ch = ch.replace("(", " ")
                    ch = ch.replace(")", " ")
                    ch = ch.strip()
                    ch = ch.split()
                    if j == 0:
                        ch = ch[1:]
                    # Take just the first time the edge appears in .anc
                    if int(ch[3]) == tree_start:
                        if int(ch[0]) != -1:
                            kk += 1
                            # edge_mutations_relate[i - 2, j] = float(ch[2])
                            if (supported_only and float(ch[2]) > 0) or (
                                not supported_only
                            ):
                                # Record edge duration
                                Left = sites[1 + int(ch[3])]
                                Right = sites[1 + int(ch[4])]
                                if Left >= pos_range[0] and Right <= pos_range[1]:
                                    edges_relate[k] = (i - 2, j)
                                    k += 1
                                    edge_durations_relate[i - 2, j] = (Left, Right)
                                    edge_lengths_relate[i - 2, j] = float(ch[1])
                    else:
                        edge_durations_relate[i - 2, j] = (-1, -1)

    print("Unique edges:", kk, "of which", k, "have at least one mutation")
    return edges_relate, edge_durations_relate, edge_lengths_relate


def run_tsinfer(ts, Ne, contig):
    """
    Run tsinfer and tsdate
    :param ts: simulated tree sequence in tskit format
    :param Ne: effective population size of DIPLOIDS
    :param contig: stdpopsim contig object
    :return:
    """
    print("Running tsinfer", flush=True)
    ts_tsinfer = tsinfer.infer(tsinfer.SampleData.from_tree_sequence(ts))
    ts_tsinfer = ts_tsinfer.simplify()

    print(
        "Trees:",
        ts_tsinfer.num_trees,
        ", samples:",
        ts_tsinfer.num_samples,
        ", sequence length:",
        ts_tsinfer.sequence_length,
        ", sites:",
        ts_tsinfer.num_sites,
        ", mutations:",
        ts_tsinfer.num_mutations,
    )
    num_muts = 0
    for s in ts_tsinfer.sites():
        if len(s.mutations) == 1:
            num_muts += 1
    print("Mutations mapping uniquely to a edge:", num_muts)
    num_nodes = 0
    num_unary = 0
    num_polytomies = 0
    for t in ts_tsinfer.trees():
        for n in t.nodes():
            num_nodes += 1
            if t.num_children(n) == 1:
                num_unary += 1
            if t.num_children(n) > 2:
                num_polytomies += 1
    print(
        "Total nodes:",
        num_nodes,
        ", unary nodes:",
        num_unary,
        ", polytomies:",
        num_polytomies,
    )

    print("Running tsdate", flush=True)
    ts_tsdate = tsdate.date(ts_tsinfer, Ne=Ne, mutation_rate=contig.mutation_rate)
    print("Done", flush=True)

    return ts_tsdate


def flat_recombination_map(recombination_rate, sequence_length):
    with open("dummy_map.txt", "w") as file:
        file.write("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n")
        file.write("0 " + str(recombination_rate * 1e8) + " 0.0\n")
        file.write(
            str(int(sequence_length))
            + " 0.0 "
            + str(recombination_rate * 1e2 * sequence_length)
            + "\n"
        )


def read_in_pop_metadata(file):
    """
    Function to read in file with 4 columns (ID, POP, GROUP, SEX) and return a dictionary with the data,
    and group and population labels
    :param file:
    :return:
    """
    metadata = {}
    groups = set()
    populations = set()
    with open(file, "r") as f:
        next(f)
        i = 0
        for line in f:
            line = line.strip().split()
            metadata[i] = {
                "ID": line[0],
                "population": line[1],
                "group": line[2],
                "sex": line[3],
            }
            metadata[i + 1] = {
                "ID": line[0],
                "population": line[1],
                "group": line[2],
                "sex": line[3],
            }
            if line[2] not in groups:
                groups.add(line[2])
            if line[1] not in populations:
                populations.add(line[1])
            i += 2
    return metadata, populations, groups


def write_poplabels(ts, filename):
    """
    Write a file with the population labels for each sample
    :param ts: simulated ts
    :param filename: where to write .poplabels file (no extension)
    :return:
    """
    with open(filename + ".poplabels", "w") as file:
        file.write("sample population group sex\n")
        for s in ts.individuals():
            node = ts.node(s.nodes[0])
            pop = ts.population(node.population).metadata["id"]
            file.write("SAM" + str(s.id) + " " + pop + " " + pop + " NA\n")
            if len(s.nodes) > 1:
                node = ts.node(s.nodes[1])
                pop_ = ts.population(node.population).metadata["id"]
                if pop_ != pop:
                    sys.exit(
                        "Warning: can't have different population IDs for the same individual."
                    )
