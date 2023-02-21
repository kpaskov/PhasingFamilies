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
- - sibpairs.json
- - inheritance_patterns
- - - [family_1].bed
- - - [family_2].bed
...

```

The `info.json` file contains metadata including the reference assembly (GRch37 or GRch38), vcf directory, ped file, sequencing error parameter files, and any flags provided to the phasing algorithm.

The `crossovers.json` and `crossovers.bed` files contains all called crossovers in the cohort in .json and .bed format respectively.

The `gene_conversions.json` and `gene_conversions.bed` files contains possible gene-conversion events in the cohort in .json and .bed format respectively. Gene conversions are more difficult to detect than crossovers due to their small size. We have not validated the gene conversions called by our algorithm, so they should be examined carefully before use.

`sibpairs.json` contains all sibling-pairs in the dataset as well as genome-wide autosomal IBD values and chromosome-level IBD values.

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
python phase/phase_chromosome.py [ped_file] [data_dir] [high_complexity_sequencing_error_profile] [low_complexity_sequencing_error_profile]
```

The script has options
- `--detect_inherited_deletions` models inherited deletions while phasing.
- `--detect_upd` models uniparental disomy while phasing. We detect only heterodisomy because isodisomy is essentially indistinguishable from a denovo deletion. (This option is still under development.)
- `--chrom [chrom]` phases a single chromosome. If this option is not used, all autosomal chromosomes are phased.
- `--family_size [family_size]` only families of size [family_size] are phased.
- `--family [family]` only [family] is phased.
- `--batch_size` and `--batch_num` options can be used to parallelize when running on a large cohort.
- `--phase_name [name]` phase data will be written to directory `[data_dir]/phase_[name]/inheritance_patterns`. If this option is not used, phase data will be written to directory `[data_dir]/phase/inheritance_patterns`.

The example below phases all autosomal chromosomes for all families of size 4 in the ssc.hg38 dataset.

```
python phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38 ../DATA/ssc.hg38/sequencing_error_rates/HCR_errors.json ../DATA/ssc.hg38/sequencing_error_rates/LCR_errors.json --detect_inherited_deletions --family_size 4 
 ```
 
 You can use as many sequencing error profiles as you'd like. This comes in handy when phasing the X chromosome, since the PAR and non-PAR regions have different error profiles. This is an example of phasing the X chromosome using four different error profiles.
 
 ```
 python phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38 ../DATA/ssc.hg38/sequencing_error_rates/X_PAR_HCR_errors.json ../DATA/ssc.hg38/sequencing_error_rates/X_PAR_LCR_errors.json ../DATA/ssc.hg38/sequencing_error_rates/X_nonPAR_HCR_errors.json ../DATA/ssc.hg38/sequencing_error_rates/X_nonPAR_LCR_errors.json --chrom X --batch_size 3 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --phase_name X
 ```

If no `--chrom` option is given, `phase_chromosome.py` will phase all autosomal chromosomes. The X chromosome must be phased explicitly using the `--chrom` option.

### 4. Pull sibling-pair IBD and run quality control metrics.
This script pulls genome-wide autosomal IBD for sibling pairs as well as chromosome-level IBD. It marks siblings who appear to be identical twins or based on their IBD sharing, and identifies siblings with unusual IBD as outliers.

```
python phase/pull_sibpair_ibd.py [data_dir]
```

The `--phase_name [name]` flag indicates that phase data from the directory `[data_dir]/phase_[name]/inheritance_patterns` will be analyzed.

This script produces `[data_dir]/phase/sibpairs.json` (or `[data_dir]/phase_[name]/sibpairs.json` if the `--phase_name` flag is used) which contains an entry for every sibling pair in the dataset with the following fields:
- `family` the family ID for the sibling pair
- `sibling1` the ID for the first sibling in the pair (the first sibling's ID is alphabetically first)
- `sibling2` the ID for the second sibling in the pair (the second sibling's ID is alphabetically last)
- `maternal_ibd` the genome-wide autosomal maternal IBD between the siblings
- `maternal_unknown_fraction` the fraction of the autosomal genome where maternal IBD could not be determined
- `maternal_ibd_chroms` a list of chromosome-level maternal IBD between the siblings in order from chr1-chr22, ending with chrX
- `maternal_unknown_fraction_chroms` a list of the fraction of each chromosome where maternal IBD could not be determined in order from chr1-chr22, ending with chrX
- `paternal_ibd` the genome-wide autosomal paternal IBD between the siblings
- `paternal_unknown_fraction` the fraction of the autosomal genome where paternal IBD could not be determined
- `paternal_ibd_chroms` a list of chromosome-level paternal IBD between the siblings in order from chr1-chr22, ending with chrX
- `paternal_unknown_fraction_chroms` a list of the fraction of each chromosome where paternal IBD could not be determined in order from chr1-chr22, ending with chrX
- `matxpat_ibd` the genome-wide autosomal IBD2 (simultaneous maternal and paternal IBD) between the siblings
- `matxpat_unknown_fraction` the fraction of the autosomal genome where IBD2 could not be determined
- `matxpat_ibd_chroms` a list of chromosome-level IBD2 between the siblings in order from chr1-chr22, ending with chrX
- `matxpat_unknown_fraction_chroms` a list of the fraction of each chromosome where IBD2 could not be determined in order from chr1-chr22, ending with chrX  
- `is_identical` siblings are marked as identical if both maternal and paternal genome-wide autosomal IBD is greater than 0.8.
- `is_ibd_outlier` siblings are marked as ibd outliers using gaussian kernel density outlier detection
```

The `qc/Visualize-Sibpair-IBD.ipynb` jupyter notebook can be used to visualize IBD in the dataset.

### 5. Pull crossovers and run quality control metrics.
This script pulls crossovers for the cohort. It identifies individuals with too many or too few crossovers as outliers.

```
python phase/pull_crossovers.py [data_dir]
```

The script has options
- `--phase_name [name]` flag indicates that phase data from the directory `[data_dir]/phase_[name]/inheritance_patterns` will be analyzed.
- `--hts_loss_regions [loss_region_index1] [loss_region_index2]` should be used if multiple sequencing error profiles are used to represent hard-to-sequence regions. For example, the X_chromosome phasing command shown in the example above has two sequencing error profiles corresponding to hard-to-sequence regions (index 1 and 3).

The script produces `[data_dir]/phase/crossovers.json` and `[data_dir]/phase/gene_conversions.json` which contain crossover and gene_conversion events respectively. Each event  has fields:
- `family` the family ID where the event was detected
- `child` the child id the event was detected within
- `chrom` the chrom where the event ocurred
- `start_pos` and `end_pos` the window where the event occured
- `is_mat` and `is_pat` indicating whether the event occured during maternal or paternal meiosis
- `is_complex` true if the event is a complex event (composed of multiple closely spaced events)

The `qc/Visualize-Crossovers.ipynb` jupyter notebook can be used to visualize crossovers in the dataset.


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

