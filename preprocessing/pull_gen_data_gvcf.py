from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import time
import gzip
from itertools import product, zip_longest
import sys

t0 = time.time()

# Pull arguments
sample_file = sys.argv[1]
chromosome = sys.argv[2]
out_directory = sys.argv[3] 

with open(sample_file, 'r') as f:
    sample_file_pairs = [line.strip().split('\t') for line in f]

print('Num individuals with genomic data', len(sample_file_pairs))

ind_chrom_to_gen = dict()

base_to_index = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 5}

# pull chromosome length from first gvcf_file header
with gzip.open(sample_file_pairs[0][1], 'rt') as f:
    chrom_length = 0
    line = next(f)
    while line.startswith('##'):
        if line.startswith('##contig=<ID=%s' % chromosome):
            length_start = line.index('length=')+7
            chrom_length = int(line[length_start:line.index(',', length_start)])
        line = next(f)
if chrom_length == 0:
    print('Chromosome length from %s not found in header.' % chromosome)


# individuals, position, read support (ref, A, C, G, T, -)
genotypes = np.zeros((len(sample_file_pairs), chrom_length, 6), dtype=np.uint8)

for i, (sample, gvcf_file) in enumerate(sample_file_pairs):
    print(i, sample)

    # read in data
    with gzip.open(gvcf_file, 'rt') as f:
        
        # Skip header
        line = next(f)
        while line.startswith('##'):
            line = next(f)
        line = next(f)

        # Loop through variants
        for line in f:
            if line.startswith(chromosome + '\t'):
                pieces = line.split('\t')
                start_pos = int(pieces[1])-1 # 0-index

                if pieces[7].startswith('END='):
                    # interval is ref, pull end_pos and dp
                    end_pos = int(pieces[7][4:])
                    dp_index = pieces[8].strip().split(':').index('DP')
                    dp = int(pieces[9].split(':', maxsplit=(dp_index+1))[dp_index])
                    genotypes[i, start_pos:end_pos, 0] = min(dp, 255) # make sure we don't overflow
                elif 'AD' in pieces[8]: # if AD doesn't exist then DP=0, and the line doesn't matter anyway
                    ref = pieces[3]
                    alt_alleles = pieces[4].split(',')[:-1]
                    ad_index = pieces[8].strip().split(':').index('AD')
                    ad = list(map(int, pieces[9].split(':', maxsplit=(ad_index+1))[ad_index].split(',')[:-1]))

                    # update ref reads
                    genotypes[i, start_pos:end_pos, 0] = min(ad[0], 255)

                    # update alt_allele reads
                    if len(alt_alleles) == 1 and len(ref) == 1 and len(alt_alleles[0]) == 1:
                        # typical SNP
                        genotypes[i, start_pos, base_to_index[alt_alleles[0][0]]] = min(ad[1], 255) # make sure we don't overflow
                    else:
                        # more complex variant
                        for alt_allele, read_count in zip(alt_alleles, ad[1:]):
                            for j, (r, a) in enumerate(zip_longest(ref, alt_allele, fillvalue='-')):
                                if r == a:
                                    genotypes[i, start_pos+j, 0] = min(genotypes[i, start_pos+j, 0]+read_count, 255) # make sure we don't overflow
                                elif r == '-':
                                    pass
                                else:
                                    genotypes[i, start_pos+j, base_to_index[a]] = min(genotypes[i, start_pos+j, base_to_index[a]]+read_count, 255)
                    
np.save('%s/chr.%s.%s' % (out_directory, chromosome, sample_file.split('/')[-1][:-12]), genotypes)
print('Completed in ', time.time()-t0, 'sec')

