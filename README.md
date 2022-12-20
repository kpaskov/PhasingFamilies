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
- inheritance_patterns
- - info.json
- - [family_1].bed
- - [family_2].bed
...
- crossovers
- - info.json
- - crossovers.json
- - crossovers.bed
- - gene_conversions.json
- - gene_conversions.bed
- - IBD.json
- deletions
- - info.json
- - inherited_deletions.json
- - inherited_deletions.bed
```

The `inheritance_patterns/info.json` file contains metadata including the reference assembly (GRch37 or GRch38), vcf directory, ped file, sequencing error parameter files, and any flags provided to the phasing algorithm.

The `[family].bed` files contain inheritance patterns for each family, across all chromosomes in .bed format.

The `crossovers.json` file contains all called crossovers in the cohort in .json format.

The `crossovers.bed` file contains all called crossovers in the cohort in .bed format.

## Instructions for running code


1. Preprocessing
You need to get your genomic data into numpy format. If your data is currently in VCF format, split by chromosome, this can be done by running

python preprocessing/pull_gen_data.py [vcf_file] [data_dir] [chrom]

If your vcf files don't have filters applied (for example no variant is PASS) or you'd like to apply a different type of filter, use preprocessing/pull_pass.py

2. Tune parameters
The phasing/deletion detection code needs estimates of different types of sequencing error rates. These parameters can be estimated automatically from the data. First, we pull out counts for joint family genotypes.

python parameter_estimation/pull_famgen_counts.py [data_dir] [ped_file] [chrom] [out_dir]

Then we estimate parameters using

python parameter_estimation/estimate_parameters.py [data_dir] [param_file]

python parameter_estimation/extend_params.py [param_file]

3. Run Phasing/Deletion Detection
Now we're ready to run using

python phase/phase_chromosome.py [chrom] [ped_file] [data_dir] [assembly version] [phase_dir] [param_file] [num_loss_regions] --detect_deletions --family AU0197


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

