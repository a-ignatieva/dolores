Citation: Ignatieva, A., Favero, M., Koskela, J., Sant, J., Myers, S.R. The length of haplotype blocks and signals of structural variation in reconstructed genealogies. bioRxiv, doi.org/10.1101/2023.07.11.548567 

## DoLoReS: Detection of Localised Recombination Suppression

To run the script, first install the necessary packages
    `pip install -r requirements.txt`
Then run
    `python -m run-dolores -h`
to see the full list of options.
The example code can be run using the provided ARG (10Mb of human chr21 simulated using stdpopsim and reconstructed using Relate):
    `python -m run-dolores -C chr21 -n chr21_100_relate -t example`
    
The input name should correspond to the location of the ARG in name.trees (or name.trees.tsz) in tskit format, name.poplabels (if using genotype IDs for clade calculations), and name.popsize (a comma-separated file with two columns, column one giving the start time of each epoch in generations, and column two the corresponding haploid population size).

The script produces plots of the results (Q-Q plot of the (one-sided) Test 1 p-values, plot of Test 1 p-value outliers, and a Manhattan plot for Test 1 and Test 2) and an output file (.csv) with the following entries:
   name: Input name of trees to process
   genetic_map: Recombination map used
   total_clades: Total number of clades in the ARG (of size > 2)
   clade_num: Index of clade
   clade_id: ID of clade
   nlog10p_test1: p-value for Test 1
   nlog10p_test2: p-value for Test 2
   cladesize: Size of clade
   span: Genomic span of clade (bp)
   start: Start position of clade
   end: End position of clade
   mut_span: Genomic span of clade measured from leftmost to rightmost supporting mutation (bp)
   left_mut: Position of leftmost supporting mutation
   right_mut: Position of rightmost supporting mutation
   num_mutations: Number of supporting mutations
   merged: How many clades have been merged together to produce this one
   chunk_index: Chunk index where clade begins
   tree_index: Tree index where clade begins
   node_id: Node ID of clade MRCA in tree where clade begins
   
## Genomic span of edges

This script computes p-values for (a subset of) edges in the input ARG. Run
    `python -m run-edgespans -h`
to see the full list of options.
The example code can be run using the provided ARG (10Mb of human chr21 simulated using stdpopsim with the SMC' model):
    `python -m run-edgespans -C chr21 -n chr21_100_smcprime -t example -T smcprime -b 1000`

The script produces a Q-Q plot and histogram of the (two-sided) p-values.