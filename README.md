# PhasingFamilies

## Purpose
This project contains code for simultaneously detecting family inheritance patterns, identity-by-descent, crossovers, and inherited deletions in nuclear families, as published in

## Input and output
This code starts with VCF files (with accompanying .tbi files), split by chromosome. It produces 
- family inheritance patterns across the genome (.BED)
- crossovers (.JSON, .BED)
- inherited deletions (.JSON, .BED)

The code adds to a directory structure created by the https://github.com/kpaskov/VCFtoNPZ project.

```
[data_dir]
- genotypes
- family_genotype_counts
- sequencing_error_rates
- phase
- - info.json
- - crossovers.json
- - crossovers.bed
- - gene_conversions.json
- - gene_conversions.bed
- - inherited_deletions.json
- - inherited_deletions.bed
- - IBD.json
- - inheritance_patterns
- - - [family_1].bed
- - - [family_2].bed
...

```

The `info.json` file contains metadata including the reference assembly (GRch37 or GRch38), vcf directory, ped file, sequencing error parameter files, and any flags provided to the phasing algorithm.

The `crossovers.json` and `crossovers.bed` files contains all called crossovers in the cohort in .json and .bed format respectively.

The `gene_conversions.json` and `gene_conversions.bed` files contains possible gene-conversion events in the cohort in .json and .bed format respectively. Gene conversions are more difficult to detect than crossovers due to their small size. We have not validated the gene conversions called by our algorithm, so they should be examined carefully before use.

`IBD.json`

The `inherited_deletions.json` and `inherited_deletions.bed` files contains all called inherited deletions in the cohort in .json and .bed format respectively.

The `[family].bed` files contain inheritance patterns for each family, across all chromosomes in .bed format.

## Instructions for running code

### 1. Start by getting your genomic data into numpy format.
using https://github.com/kpaskov/VCFtoNPZ. 

### 2. Estimate sequencing error rates.
using https://github.com/kpaskov/FamilySeqError.

If using whole-genome sequencing data, follow the instructions for estimating sequencing error rates in both low-complexity and high-complexity regions. A good source of low-complexity regions is the supplementary materials file from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4271055/.

### 3. Run our hidden Markov model family phasing algorithm.
The algorithm detects crossovers and inherited deletions. Run using

```
python phase/phase_chromosome.py [ped_file] [data_dir] [high_complexity_param_file] [low_complexity_param_file] --detect_inherited_deletions
```

The `--batch_size` and `--batch_num` options can be used to parallelize when running on a large cohort.

---------------------------------------------------------------------------------------------------------

Memory
WGS: families of size 3/4 need 8GB, families of size 5 need 16GB, families of size 6 need 64GB
Exome: families of size 3/4/5/6/7 need 8GB

family batch:

for familysize in 3 4 5 6 7
do
   python phase/phase_chromosome.py 2 $familysize data/ancestry.ped split_gen_ancestry 37 phased_ancestry parameter_estimation/params/ancestry_params_ext.json FALSE
done

1. phase

python phase_chromosome.py

2. pull sibpair similarity with jupyter notebook

sibpair_similarity/Pull-Twins.ipynb

3. pull recombination

python phase/pull_crossovers.py phased_ssc.hg38_phase1-1_upd

4. QC sibpairs with jupyter notebook

phase/IBD.ipynb

5. pull recombination

python phase/pull_crossovers.py recomb_ssc.hg38_upd

6. pull deletions

python phase/proccess_deletions.py recomb_ssc.hg38_upd

7. pull upd

python phase/pull_upd.py recomb_ssc.hg38_upd

