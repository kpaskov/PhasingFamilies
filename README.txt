This project contains code for simultaneously phasing and detecting deletions in nuclear families.

1. Preprocessing
You need to get your genomic data into numpy format. If your data is currently in VCF format, split by chromosome, this can be done by running

python preprocessing/pull_gen_data.py [vcf_file] [data_dir] [chrom]

If your vcf files don't have filters applied (for example no variant is PASS) or you'd like to apply a different type of filter, use preprocessing/pull_pass.py

2. Tune parameters
The phasing/deletion detection code needs estimates of different types of sequencing error rates. These parameters can be estimated automatically from the data. First, we pull out counts for joint family genotypes.

python parameter_estimation/pull_famgen_counts.py [data_dir] [ped_file] [chrom]

Then we estimate parameters using

python parameter_estimation/estimate_parameters_per_individual.py [data_dir] [param_file]

3. Run Phasing/Deletion Detection
Now we're ready to run using

python phase/phase_chromosome.py [chrom] [family_size] [ped_file] [data_dir] [assembly version] [phase_dir] [param_file] [detect denovos]

Memory
WGS: families of size 3/4 need 8GB, families of size 5 need 16GB, families of size 6 need 64GB
Exome: families of size 3/4/5/6/7 need 8GB