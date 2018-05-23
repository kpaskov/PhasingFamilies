from collections import defaultdict
import numpy as np
from scipy.sparse import csc_matrix, save_npz
import time
import gzip
from itertools import product
import sys

t0 = time.time()

# Pull arguments
vcf_file = sys.argv[1]
ped_file = sys.argv[2]
out_directory = sys.argv[3]
chrom = sys.argv[4]
cutoff = int(sys.argv[5])

# Pull data from vcf
with gzip.open(vcf_file, 'rt') as f, \
    open('%s/chr.%s.dp.samples.txt' % (out_directory, chrom), 'w+') as sample_f, \
    gzip.open('%s/chr.%s.dp.variants.txt.gz' % (out_directory, chrom), 'wt') as variant_f:

    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull sample_ids and write to file
    sample_ids = line.strip().split('\t')[9:]
    sample_f.write('\n'.join(sample_ids))
    print('Num individuals with genomic data', len(sample_ids))

    # Pull genotypes from vcf
    m = len(sample_ids)
    data, indices, indptr = [], [], [0]
    
    line = next(f)
    subfile = 0
    num_lines = 0
    for j, line in enumerate(f):
        pieces = line.split('\t')

        # Write variant to file
        variant_f.write('\t'.join(pieces[:9]) + '\n')

        # Pull out genotypes
        format = pieces[8].strip().split(':')
        gen_index = format.index('DP')
        for i, piece in enumerate(pieces[9:]):
            segment = piece.split(':', maxsplit=gen_index+1)

            dp = int(segment[gen_index])
            if dp <= cutoff:
                indices.append(i)
                data.append(dp)
        indptr.append(len(data))
        num_lines += 1

        # If file has gotten really big, write subfile to disk
        if len(data) > 500000000:
            dp = csc_matrix((data, indices, indptr), shape=(m, num_lines), dtype=np.int8)
            print('Sub dataset', dp.shape)

            # Save to file
            save_npz('%s/chr.%s.%d.dp' % (out_directory, chrom, subfile), dp)

            # Start fresh
            subfile += 1
            num_lines = 0
            data, indices, indptr = [], [], [0]

    dp = csc_matrix((data, indices, indptr), shape=(m, num_lines), dtype=np.int8)
    
    # Save to file
    if subfile == 0:
        print('Full dataset', dp.shape)
        save_npz('%s/chr.%s.dp' % (out_directory, chrom), dp)
    else:
        print('Sub dataset', dp.shape)
        save_npz('%s/chr.%s.%d.dp' % (out_directory, chrom, subfile), dp)

print('Completed in ', time.time()-t0, 'sec')

