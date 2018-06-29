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
    open('%s/chr.%s.ad.samples.txt' % (out_directory, chrom), 'w+') as sample_f, \
    gzip.open('%s/chr.%s.ad.variants.txt.gz' % (out_directory, chrom), 'wt') as variant_f:

    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull sample_ids and write to file
    sample_ids = line.strip().split('\t')[9:]
    sample_f.write('\n'.join(sample_ids))
    print('Num individuals with genomic data', len(sample_ids))

    # Pull AD from vcf
    m = len(sample_ids)
    data, indices, indptr = [], [], [0]
    
    line = next(f)
    subfile = 0
    num_lines = 0
    for j, line in enumerate(f):
        pieces = line.split('\t')

        # Write variant to file
        variant_f.write('\t'.join(pieces[:9]) + '\n')

        # Pull out AD
        format = pieces[8].strip().split(':')
        ad_index = format.index('AD')
        for i, piece in enumerate(pieces[9:]):
            segment = piece.split(':', maxsplit=ad_index+1)

            ads = -2, -2
            if segment[ad_index] != '.':
                try:
                    ads = tuple(map(int, segment[ad_index].split(',')))[:2]
                except:
                    print(format, ad_index, segment)

            if ad[0] <= cutoff:
                indices.append(2*i)
                data.append(ad[0]+1)
            if ad[1] <= cutoff:
                indices.append((2*i)+1)
                data.append(ad[1]+1)

        indptr.append(len(data))
        num_lines += 1

        # If file has gotten really big, write subfile to disk
        if len(data) > 500000000:
            ad = csc_matrix((data, indices, indptr), shape=(m, num_lines), dtype=np.int8)
            print('Sub dataset', ad.shape)

            # Save to file
            save_npz('%s/chr.%s.%d.ad.cutoff.%d' % (out_directory, chrom, subfile, cutoff), ad)

            # Start fresh
            subfile += 1
            num_lines = 0
            data, indices, indptr = [], [], [0]

    ad = csc_matrix((data, indices, indptr), shape=(m, num_lines), dtype=np.int8)
    
    # Save to file
    if subfile == 0:
        print('Full dataset', ad.shape)
        save_npz('%s/chr.%s.ad.cutoff.%d' % (out_directory, chrom, cutoff), ad)
    else:
        print('Sub dataset', ad.shape)
        save_npz('%s/chr.%s.%d.ad.cutoff.%d' % (out_directory, chrom, subfile, cutoff), ad)

print('Completed in ', time.time()-t0, 'sec')

