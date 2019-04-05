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

gen_mapping = {'./.': -1, '0/0': 0, '0|0': 0, '0/1': 1, '0|1': 1, '1/0': 1, '1|0': 1, '1/1': 2, '1|1': 2}

counts = np.zeros((4, 101), dtype=int)

# Pull data from vcf
with gzip.open(vcf_file, 'rt') as f:
    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull sample_ids
    sample_ids = line.strip().split('\t')[9:]
    m = len(sample_ids)
    print('Num individuals with genomic data', m)

    # Pull genotypes/depth from vcf
    line = next(f)
    for j, line in enumerate(f):
        pieces = line.split('\t')

        # Pull out genotypes/depth
        format = pieces[8].strip().split(':')
        gen_index = format.index('GT')
        dp_index = format.index('DP')
        max_index = max(gen_index, dp_index)+1
        for i, piece in enumerate(pieces[9:]):
            segment = piece.split(':', maxsplit=max_index)

            if segment[dp_index] != '.' and segment[gen_index] in gen_mapping:
                try:
                    dp = int(segment[dp_index])
                    gt = gen_mapping[segment[gen_index]]

                    if dp > 100:
                        dp == 100
                    counts[gt, dp] += 1
                except:
                    print(format, piece)

           


        # Pull out genotypes
            gen_index = format.index('GT')
            for i, piece in enumerate(pieces[9:]):
                segment = piece.split(':', maxsplit=gen_index+1)
                if segment[gen_index] in gen_mapping:
                    gt = gen_mapping[segment[gen_index]]
                    if gt == -1:
                        if 'DP' in format:
                            dp_index = format.index('DP')
                            segment = piece.split(':', maxsplit=dp_index+1)
                            if dp_index < len(segment) and (segment[dp_index] == '0' or segment[dp_index] == '1'):
                                # very low coverage is marked double deletion rather than unknown
                                gt = -2
                else:
                    # For now we mark multi-base loci as unknown
                    gt = -1

                if gt != 0:
                    indices.append(i)
                    data.append(gt)

        indptr.append(len(data))
        num_lines += 1

        # If file has gotten really big, write subfile to disk
        if len(data) > 500000000:
            dp = csc_matrix((data, indices, indptr), shape=(m, num_lines), dtype=np.int8)
            print('Sub dataset', dp.shape)

            # Save to file
            save_npz('%s/chr.%s.%d.dp.cutoff.%d' % (out_directory, chrom, subfile, cutoff), dp)

            # Start fresh
            subfile += 1
            num_lines = 0
            data, indices, indptr = [], [], [0]

    dp = csc_matrix((data, indices, indptr), shape=(m, num_lines, cutoff), dtype=np.int8)
    


print('Completed in ', time.time()-t0, 'sec')

