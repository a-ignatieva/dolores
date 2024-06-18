#!/usr/bin/env python
# coding: utf-8

import sys
import os
import getopt
import gc
import pickle
from tqdm import tqdm
import tskit
import tszip
import msprime
import stdpopsim
import dolores.cladespans
import dolores.simulations
import dolores.viz


def main(argv):
    chromosome = None
    name = None
    trees_loc = None
    species = "HomSap"
    Ne_file = None
    genetic_map_stdpopsim = "HapMapII_GRCh37"
    genetic_map_loc = None
    use_genotype_id = True
    overwrite = False
    cM_limit = 0.01
    muts_per_kb = 0.05
    use_muts = True
    plot_on = True
    try:
        opts, args = getopt.getopt(
            argv,
            "hC:n:t:s:G:g:u:c:m:M:op",
            [
                "chromosome=",
                "name=",
                "trees_loc=",
                "species_stdpopsim=",
                "genetic_map_stdpopsim=",
                "genetic_map_loc=",
                "use_genotype_id=",
                "cM_limit=",
                "muts_per_kb=",
                "mut_spans=",
                "overwrite",
                "plot_off",
            ],
        )
    except getopt.GetoptError:
        print("For usage: python -m run-dolores -h")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("")
            print("python -m run-dolores -C <chromosome> -n <name> -t <filepath> ")
            print("-" * 100)
            print("Example:")
            print("python -m run-dolores -C chr21 -n chr21_100_relate -t example")
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
                "-u",
                "--use_genotype_id",
                ": Whether to use genotype ID (or sample ID; if yes, must have a .poplabels file",
                "\n                       in the same directory as the trees; default: 1), 0 or 1",
            )
            print(
                "-c",
                "--cM_limit",
                ": How far apart can clades be for them to be merged together (default: 0.01), float",
            )
            print(
                "-m",
                "--muts_per_kb",
                ": Required number of mutations per kb, float (default: 0.05)",
            )
            print(
                "-M",
                "--mut_spans",
                ": Whether to use mutations to calculate clade span, 0 or 1 (default: 1)",
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
        elif opt in ("-u", "--use_genotype_id"):
            use_genotype_id = bool(int(arg))
        elif opt in ("-c", "--cM_limit"):
            cM_limit = float(arg)
        elif opt in ("-m", "--muts_per_kb"):
            muts_per_kb = float(arg)
        elif opt in ("-M", "--mut_spans"):
            use_muts = bool(int(arg))
        elif opt in ("-o", "--overwrite"):
            overwrite = True
        elif opt in ("-p", "--plot"):
            plot_on = False

    if (
        chromosome is None
        or trees_loc is None
        or (genetic_map_loc is None and genetic_map_stdpopsim is None)
    ):
        print("For usage: python -m run-dolores -h")
        sys.exit(2)

    if not os.path.exists(trees_loc + "/" + name + ".popsize"):
        print("Error: must have a .popsize file")
        sys.exit(2)

    if use_genotype_id and not os.path.exists(trees_loc + "/" + name + ".poplabels"):
        print("Warning: option set to use genotype IDs, but .poplabels file not found")
        print("Defining clades through sample IDs instead")
        use_genotype_id = False

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
            print("Must specify recombination map file or species")
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
    print("Recombination map from stdpopsim:", genetic_map_stdpopsim)
    print("Recombination map location:", genetic_map_loc)
    print("Using genotype IDs:", use_genotype_id)
    print("cM limit for merging clades:", cM_limit)
    print("Mutations per kb:", muts_per_kb)
    print("Overwrite:", overwrite)
    print("=" * 100)

    output_dir = trees_loc + "/" + name + "_output"
    if not os.path.exists(output_dir):
        print("Making results directory: " + output_dir)
        os.mkdir(output_dir)
    print("-" * 100)

    # ----------------------------------------------------------------------------------------------------------------------
    # GETTING METADATA
    # ----------------------------------------------------------------------------------------------------------------------

    if (
        os.path.exists(output_dir + "/" + name + ".clades.gz")
        and os.path.exists(output_dir + "/" + name + ".pvalues.gz")
        and not overwrite
    ):
        print("Results already exist, loading. Specify -o option to overwrite.")
        results = dolores.cladespans.read_from_file(output_dir + "/" + name)
    else:
        if use_genotype_id:
            print("Getting metadata...")
            md, populations, groups = dolores.simulations.read_in_pop_metadata(
                trees_loc + "/" + name + ".poplabels"
            )
            print("Groups and populations:")
            print(groups, populations)

            filename = output_dir + "/" + name + "_samp_to_ind.pickle"
            if not os.path.exists(filename):
                print("Getting map of samples to individuals...")
                samp_to_ind = {}
                inds = {}
                i = 0
                for key, val in md.items():
                    if val["ID"] not in inds:
                        inds[val["ID"]] = i
                        i += 1
                    samp_to_ind[key] = inds[val["ID"]]
                with open(filename, "wb") as file:
                    pickle.dump(samp_to_ind, file)
                del samp_to_ind
                gc.collect()
        else:
            samp_to_ind = None

        # ------------------------------------------------------------------------------------------------------------------
        # SPLITTING UP TREES
        # ------------------------------------------------------------------------------------------------------------------

        print("Splitting up trees for " + name + "...")
        filename_trees = trees_loc + "/" + name + ".trees"
        if os.path.exists(filename_trees + ".tsz"):
            ts = tszip.decompress(filename_trees + ".tsz")
        elif os.path.exists(filename_trees):
            ts = tskit.load(filename_trees)
        else:
            sys.exit("No tree sequence file found in " + output_dir)

        num_samples = ts.num_samples
        num_trees = ts.num_trees
        bps = ts.breakpoints(as_array=True)
        num_chunks = int(len(bps) / 1000) + 1

        for i, j in enumerate(range(num_chunks)):
            filename_chunk = output_dir + "/" + name + "_chunk" + str(j) + ".trees"
            if not os.path.exists(filename_chunk) and not os.path.exists(
                filename_chunk + ".tsz"
            ):
                print("...chunk", j)
                left = bps[i * 1000]
                if (i + 1) * 1000 >= len(bps):
                    right = bps[-1]
                else:
                    right = bps[(i + 1) * 1000]
                ts_sub = ts.keep_intervals([[left, right]])
                tszip.compress(ts_sub, filename_chunk + ".tsz")
                del ts_sub
                gc.collect()
            else:
                print("...chunk", j, "already done")
        del ts
        gc.collect()
        print("Done")
        print("-" * 100)

        if not os.path.exists(output_dir + "/" + name + "_treeinfo.txt"):
            print("Getting tree info...")
            with open(output_dir + "/" + name + "_treeinfo.txt", "w") as file:
                file.write("chr;chunk;chunk_tree_index;tree_start;tree_end\n")
                for i in range(num_chunks):
                    print("Loading chunk " + str(i) + "...")
                    filename = output_dir + "/" + name + "_chunk" + str(i) + ".trees"
                    if os.path.exists(filename + ".tsz"):
                        ts = tszip.decompress(filename + ".tsz")
                    else:
                        ts = tskit.load(filename)
                    for t in ts.trees():
                        if t.num_roots == 1:
                            file.write(
                                name
                                + ";"
                                + str(i)
                                + ";"
                                + str(t.index)
                                + ";"
                                + str(int(t.interval[0]))
                                + ";"
                                + str(int(t.interval[1]))
                                + "\n"
                            )

        print("-" * 100)

        filename = output_dir + "/" + name + "_unmerged"
        if not os.path.exists(filename + ".clades.gz"):
            if use_genotype_id:
                samp_to_ind = pickle.load(
                    open(output_dir + "/" + name + "_samp_to_ind.pickle", "rb")
                )
            ts_handles = [
                output_dir + "/" + name + "_chunk" + str(i) + ".trees"
                for i in range(num_chunks)
            ]
            _, duplicate_clades = dolores.cladespans.clade_span(
                ts_handles,
                num_trees,
                num_samples,
                samp_to_ind=samp_to_ind,
                write_to_file=filename,
                write_to_file_freq=50000,
            )

            print("-" * 100)
            print("Clades that are duplicates based on genotype ID:")
            print(
                "chunk;chunk_tree_index;tree_index;node_id;cladesize;sample_ids;individual_ids"
            )
            for d in duplicate_clades:
                print(*d, sep=";")
            print("-" * 100)

            del ts_handles
            gc.collect()
        else:
            print("Calculating clade spans already done")

        print("-" * 100)

        # ------------------------------------------------------------------------------------------------------------------
        # MERGING CLADES
        # ------------------------------------------------------------------------------------------------------------------

        filename = output_dir + "/" + name
        if not os.path.exists(filename + ".clades.gz"):
            filename_ = output_dir + "/" + name + "_unmerged"
            print("Reading in unmerged clades...")
            results = dolores.cladespans.read_from_file(filename_)
            results.merge_clades(recombination_map, cM_limit=cM_limit)
            print("Done - storing merged results...")
            results.write_to_file(filename)
        else:
            results = dolores.cladespans.read_from_file(filename)

        results_chunk = []
        chunk = 0
        with tqdm(
            total=results.num, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
        ) as pbar:
            for i in range(results.num):
                if results.on[i] == 1:
                    if results.chunkindex[i] != chunk:
                        with open(
                            output_dir
                            + "/"
                            + name
                            + "_chunk"
                            + str(chunk)
                            + ".results.pickle",
                            "wb",
                        ) as file:
                            pickle.dump(results_chunk, file)
                        results_chunk = []
                        chunk = results.chunkindex[i]
                    results_chunk.append(
                        [
                            results.id[i],
                            results.nodeid[i],
                            results.treeindex[i],
                            results.tbl[i],
                            results.mut_span[i],
                            results.left_mut[i],
                            results.right_mut[i],
                            results.span[i],
                            results.start[i],
                            results.end[i],
                            results.num_mutations[i],
                        ]
                    )
                pbar.update(1)

        with open(
            output_dir + "/" + name + "_chunk" + str(chunk) + ".results.pickle",
            "wb",
        ) as file:
            pickle.dump(results_chunk, file)

        num_chunks = chunk + 1

        # Free up RAM and will reload later
        del results
        del results_chunk
        gc.collect()

        print("-" * 100)

        # ------------------------------------------------------------------------------------------------------------------
        # COMPUTING p-VALUES
        # ------------------------------------------------------------------------------------------------------------------

        print(num_chunks)
        for chunk in range(num_chunks):
            filename = (
                output_dir + "/" + name + "_chunk" + str(chunk) + ".cladesinfo.pickle"
            )
            if not os.path.exists(filename):
                print("Loading chunk " + str(chunk) + "...")
                results_chunk = pickle.load(
                    open(
                        output_dir
                        + "/"
                        + name
                        + "_chunk"
                        + str(chunk)
                        + ".results.pickle",
                        "rb",
                    )
                )
                filename_ = output_dir + "/" + name + "_chunk" + str(chunk) + ".trees"
                if os.path.exists(filename_ + ".tsz"):
                    ts_chunk = tszip.decompress(filename_ + ".tsz")
                else:
                    ts_chunk = tskit.load(filename_)
                trees_chunk = dolores.edgespans.compute_trees(
                    ts_chunk, epoch_starts, Ne_list
                )
                print("Computing clade p-values for chunk " + str(chunk) + "...")
                cladesInfo = dolores.cladespans.calculate_q(
                    results_chunk,
                    recombination_map,
                    ts_chunk,
                    trees_chunk,
                    bps,
                    use_muts=use_muts,
                    muts_per_kb=muts_per_kb,
                    destroy_trees=True,
                )

                with open(filename, "wb") as file:
                    pickle.dump(cladesInfo, file)

                del results_chunk
                del ts_chunk
                del trees_chunk
                del cladesInfo
                gc.collect()
            else:
                print("Chunk", chunk, "already done.")

        print("Loading clade span results...")
        results = dolores.cladespans.read_from_file(output_dir + "/" + name)
        results.reset()

        for chunk in range(num_chunks):
            print("Loading results chunk " + str(chunk) + "...")
            results_chunk = pickle.load(
                open(
                    output_dir
                    + "/"
                    + name
                    + "_chunk"
                    + str(chunk)
                    + ".cladesinfo.pickle",
                    "rb",
                )
            )
            print("Adding chunk to results...")
            dolores.cladespans.add_info(results, results_chunk)

        print("Saving...")
        results.write_to_file(output_dir + "/" + name)

        # Tidy up
        if os.path.exists(filename):
            chunk_filenames = output_dir + "/" + name + "_chunk*.results.pickle"
            os.system("rm " + chunk_filenames)
            chunk_filenames = output_dir + "/" + name + "_chunk*.cladesinfo.pickle"
            os.system("rm " + chunk_filenames)

        print("=" * 100)

    # ----------------------------------------------------------------------------------------------------------------------
    # OUTPUTTING RESULTS
    # ----------------------------------------------------------------------------------------------------------------------

    print("Outputting results...")
    with open(output_dir + "/" + name + "_output.csv", "w") as file:
        file.write(
            "name,genetic_map,total_clades,clade_num,clade_id,"
            + "nlog10p_test1,nlog10p_test2,cladesize,span,start,end,mut_span,"
            + "left_mut,right_mut,num_mutations,merged,chunk_index,tree_index,node_id\n"
        )
        for i, clade_id in enumerate(results.ids):
            file.write(
                name
                + ","
                + "HapMapII_GRCh37"
                + ","
                + str(results.num)
                + ","
                + str(clade_id)
                + ","
                + str(i)
                + ","
                + str(-results.log10sf[i])
                + ","
                + str(-results.log10sf_event[i])
                + ","
                + str(results.cladesize[clade_id])
                + ","
                + str(int(results.span[clade_id]))
                + ","
                + str(int(results.start[clade_id]))
                + ","
                + str(int(results.end[clade_id]))
                + ","
                + str(int(results.mut_span[clade_id]))
                + ","
                + str(int(results.left_mut[clade_id]))
                + ","
                + str(int(results.right_mut[clade_id]))
                + ","
                + str(results.num_mutations[clade_id])
                + ","
                + str(results.merged[clade_id])
                + ","
                + str(results.chunkindex[clade_id])
                + ","
                + str(results.treeindex[clade_id])
                + ","
                + str(results.nodeid[clade_id])
                + "\n",
            )

    # ----------------------------------------------------------------------------------------------------------------------
    # PLOTS
    # ----------------------------------------------------------------------------------------------------------------------

    if plot_on:
        print("Plotting results...")
        dolores.viz.qqplot(
            [results],
            edges=False,
            legend_labels=["Sim, no inv (Relate)"],
            save_to_file=output_dir + "/" + name + "_qq.png",
        )
        dolores.viz.outliers_plot(
            results,
            save_to_file=output_dir + "/" + name + "_outliers.png",
        )
        dolores.viz.pvalues_plot(
            results,
            save_to_file=output_dir + "/" + name + "_pvalues.png",
        )
    print("=" * 100)


if __name__ == "__main__":
    main(sys.argv[1:])
