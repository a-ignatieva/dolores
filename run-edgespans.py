#!/usr/bin/env python
# coding: utf-8

import sys
import os
import getopt
import tskit
import tszip
import msprime
import stdpopsim
import dolores.edgespans
import dolores.simulations
import dolores.viz


def main(argv):
    chromosome = None
    name = None
    trees_loc = None
    species = "HomSap"
    genetic_map_stdpopsim = "HapMapII_GRCh38"
    genetic_map_loc = None
    overwrite = False
    plot_on = True
    mutation_rate = None
    num_sample_edges = None
    argtype = "smcprime"
    try:
        opts, args = getopt.getopt(
            argv,
            "hC:n:t:s:G:g:T:m:b:op",
            [
                "chromosome=",
                "name=",
                "trees_loc=",
                "species_stdpopsim=",
                "genetic_map_stdpopsim=",
                "genetic_map_loc=",
                "mutation_rate=",
                "num_sample_edges=",
                "argtype=",
                "overwrite",
                "plot_off",
            ],
        )
    except getopt.GetoptError:
        print("For usage: python -m run-edgespans -h")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("")
            print("python -m run-edgespans -C <chromosome> -n <name> -t <filepath> ")
            print("-" * 100)
            print("Example:")
            print(
                "python -m run-edgespans -C chr21 -n chr21_100_smcprime -t example -T smcprime -b 1000"
            )
            print("-" * 100)
            print("Options:")
            print("-" * 100)
            print("-C", "--chromosome", ": Chromosome (e.g. chr21), str")
            print("-n", "--name", ": Name of trees file (without extension), str")
            print(
                "-t",
                "--trees_loc",
                ": Location of trees file (output directory will be created here), str",
            )
            print(
                "-s",
                "--species",
                ": Species in stdpopsim catalogue (for getting Ne and genetic map), str",
            )
            print(
                "-b",
                "--num_sample_edges",
                ": Number of edges to sample, int",
            )
            print(
                "-G",
                "--genetic_map_stdpopsim",
                ": Name of genetic map to use from stdpopsim (default: HapMapII_GRCh37), str",
            )
            print(
                "-g",
                "--genetic_map_loc",
                ": Location of genetic map to load from file, str",
            )
            print(
                "-T",
                "--argtype",
                ": Specify argtype of ARG: smcprime, relate, tsdate, argneedle, argweaver, cwr",
            )
            print(
                "-m",
                "--mutation_rate",
                ": Required for Relate trees, float",
            )
            print(
                "-o",
                "--overwrite",
                ": If output files already exist, whether to recompute p-values (default: False)",
            )
            print(
                "-p",
                "--plot_off",
                ": Whether to skip plotting the results (default: False)",
            )
            print("-" * 100)
            sys.exit()
        elif opt in ("-C", "--chromosome"):
            chromosome = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-t", "--trees_loc"):
            trees_loc = arg
        elif opt in ("-s", "--species"):
            species = arg
        elif opt in ("-G", "--genetic_map_stdpopsim"):
            genetic_map_stdpopsim = arg
        elif opt in ("-g", "--genetic_map_loc"):
            genetic_map_loc = arg
        elif opt in ("-b", "--num_sample_edges"):
            num_sample_edges = int(arg)
        elif opt in ("-T", "--argtype"):
            argtype = arg
        elif opt in ("-m", "--mutation_rate"):
            mutation_rate = float(arg)
        elif opt in ("-o", "--overwrite"):
            overwrite = True
        elif opt in ("-p", "--plot_off"):
            plot_on = False

    if (
        chromosome is None
        or trees_loc is None
        or (genetic_map_loc is None and genetic_map_stdpopsim is None)
    ):
        print("For usage: python -m run-edgespans -h")
        sys.exit(2)

    if not os.path.exists(trees_loc + "/" + name + ".popsize"):
        print("Error: must have a .popsize file")
        sys.exit(2)

    epoch_starts = []
    Ne_list = []
    Ne_file = trees_loc + "/" + name + ".popsize"
    with open(Ne_file, "r") as file:
        for line in file:
            line = line.strip().split(",")
            if line[1] != "NA":
                epoch_starts.append(float(line[0]))
                Ne_list.append(float(line[1]))
            else:
                epoch_starts.append(1000000000.0)

    if genetic_map_loc is None:
        if species is None:
            print("Must specify species")
            sys.exit(2)
        if genetic_map_loc is None:
            print("Getting recombination map from stdpopsim...")
            species_ = stdpopsim.get_species(species)
            contig = species_.get_contig(
                chromosome=chromosome, genetic_map=genetic_map_stdpopsim
            )
            recombination_map = contig.recombination_map
    else:
        print("Getting recombination map from file...")
        recombination_map = msprime.RateMap.read_hapmap(genetic_map_loc)

    print("=" * 100)
    print("Inputs:")
    print("Chromosome:", chromosome)
    print("Trees location:", trees_loc)
    print("Species:", species)
    print("ARG type:", argtype)
    print("Recombination map from stdpopsim:", genetic_map_stdpopsim)
    print("Recombination map location:", genetic_map_loc)
    print("Mutation rate:", mutation_rate)
    print("Number of edges to sample:", num_sample_edges)
    print("Overwrite:", overwrite)
    print("=" * 100)

    output_dir = trees_loc + "/" + name + "_output"
    if not os.path.exists(output_dir):
        print("Making results directory: " + output_dir)
        os.mkdir(output_dir)
        print("-" * 100)

    # ----------------------------------------------------------------------------------------------------------------------
    # CALCULATIONS
    # ----------------------------------------------------------------------------------------------------------------------

    if os.path.exists(output_dir + "/" + name + ".edges.gz") and not overwrite:
        print("Results already exist, loading. Specify -o option to overwrite.")
        results = dolores.edgespans.read_from_file(output_dir + "/" + name)
    else:
        print("Loading trees...")
        filename_trees = trees_loc + "/" + name + ".trees"
        if os.path.exists(filename_trees + ".tsz"):
            ts = tszip.decompress(filename_trees + ".tsz")
        elif os.path.exists(filename_trees):
            ts = tskit.load(filename_trees)
        else:
            sys.exit("No tree sequence file found in " + output_dir)

        top_only = False
        supported_only = False
        tsdate_trees = False
        relate_trees = argw_trees = argn_trees = argn_norm = cwr_trees = False
        relate_anc_file = None
        if argtype == "relate":
            relate_trees = True
            relate_anc_file = trees_loc + "/" + name
            supported_only = True
            if mutation_rate is None:
                sys.exit("Error: need to supply mutation rate for Relate trees.")
        elif argtype == "tsdate":
            tsdate_trees = True
            top_only = True
        elif argtype == "argneedle":
            argn_trees = True
            argn_norm = False
        elif argtype == "argweaver":
            argw_trees = True
        elif argtype == "cwr":
            cwr_trees = True

        results = dolores.edgespans.gof_calc(
            ts,
            recombination_map,
            epoch_starts,
            Ne_list,
            top_only=top_only,
            supported_only=supported_only,
            mutation_rate=mutation_rate,
            tsdate_trees=tsdate_trees,
            relate_trees=relate_trees,
            relate_anc_file=relate_anc_file,
            argw_trees=argw_trees,
            argn_trees=argn_trees,
            cwr_trees=cwr_trees,
            num_sample_edges=num_sample_edges,
            write_to_file=output_dir + "/" + name,
        )

    # ----------------------------------------------------------------------------------------------------------------------
    # PLOTS
    # ----------------------------------------------------------------------------------------------------------------------

    if plot_on:
        print("Plotting results...")
        dolores.viz.qqplot(
            [results],
            edges=True,
            save_to_file=output_dir + "/" + name + "_edges_qq.png",
        )
        dolores.viz.qqplot(
            [results],
            hist_plot=True,
            edges=True,
            save_to_file=output_dir + "/" + name + "_edges_hist.png",
        )
    print("=" * 100)


if __name__ == "__main__":
    main(sys.argv[1:])
