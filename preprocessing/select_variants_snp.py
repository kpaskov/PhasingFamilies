import sys
from os import listdir

import gzip

import numpy as np
from scipy import sparse
from scipy.stats import chi2_contingency

# run with python phase/select_variants_snp.py 22 split_gen_miss

chrom = sys.argv[1]
data_dir = sys.argv[2]

variant_file = '%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom)

# pull snp indices and positions
snp_indices = []
snp_positions = []
with gzip.open(variant_file, 'rt') as f:
    for i, line in enumerate(f):
        pieces = line.strip().split('\t')
        if len(pieces[3]) == 1 and len(pieces[4]) == 1 and pieces[3] != '.' and pieces[4] != '.':
            snp_indices.append(i)
        snp_positions.append(int(pieces[1]))
snp_positions = np.array(snp_positions)

num_clean_indices = 0
with open('%s/clean_indices_%s.txt' % (data_dir, chrom), 'w+') as f:
	for snp_index in snp_indices:
		f.write('%d\t%d\n' % (snp_index, snp_positions[snp_index]))
		num_clean_indices += 1

print('Num clean indices', num_clean_indices)
