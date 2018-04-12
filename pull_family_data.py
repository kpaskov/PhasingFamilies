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

# Pull data from vcf
with gzip.open(vcf_file, 'rt') as f, \
    open('%s/chr.%s.gen.samples.txt' % (out_directory, chrom), 'w+') as sample_f, \
    gzip.open('%s/chr.%s.gen.variants.txt.gz' % (out_directory, chrom), 'wt') as variant_f:

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
    gen_mapping = {'./.': -1, '0/0': 0, '0|0': 0, '0/1': 1, '0|1': 1, '1/0': 1, '1|0': 1, '1/1': 2, '1|1': 2}
    
    line = next(f)
    for j, line in enumerate(f):
        pieces = line.split('\t')

        # Write variant to file
        variant_f.write('\t'.join(pieces[:9]) + '\n')

        # Pull out genotypes
        format = pieces[8].strip().split(':')
        gen_index = format.index('GT')
        for i, piece in enumerate(pieces[9:]):
            segment = piece.split(':', maxsplit=gen_index+1)

            if segment[gen_index] in gen_mapping:
                gt = gen_mapping[segment[gen_index]]
            else:
                # For now we mark multi-base loci as unknown
                gt = -1

            if gt != 0:
                indices.append(i)
                data.append(gt)
        indptr.append(len(data))

    n = j+1
    gen = csc_matrix((data, indices, indptr), shape=(m, n), dtype=np.int8)
    print('Full dataset', gen.shape)

    # Save to file
    save_npz('%s/chr.%s.gen' % (out_directory, chrom), gen)

print('Completed in ', time.time()-t0, 'sec')

