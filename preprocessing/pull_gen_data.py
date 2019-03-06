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
out_directory = sys.argv[2]
chrom = sys.argv[3]

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
    subfile = 0
    num_lines = 0
    chrom_coord = []
    for j, line in enumerate(f):
        pieces = line.split('\t')

        if pieces[0] == chrom:

            format = pieces[8].strip().split(':')

            # Write variant to file
            variant_f.write('\t'.join(pieces[:9]) + '\n')

            pos, _, ref, alt = pieces[1:5]
            is_biallelic_snp = 1 if len(ref) == 1 and len(alt) == 1 and ref != '.' and alt != '.' else 0
            chrom_coord.append((0, int(pos), is_biallelic_snp))

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
                gen = csc_matrix((data, indices, indptr), shape=(m, num_lines), dtype=np.int8)
                print('Sub dataset', gen.shape)

                # Save to file
                save_npz('%s/chr.%s.%d.gen' % (out_directory, chrom, subfile), gen)

                # Start fresh
                subfile += 1
                num_lines = 0
                data, indices, indptr = [], [], [0]
           

gen = csc_matrix((data, indices, indptr), shape=(m, num_lines), dtype=np.int8)
    
# Save to file
if subfile == 0:
    print('Full dataset', gen.shape)
    save_npz('%s/chr.%s.gen' % (out_directory, chrom), gen)
else:
    print('Sub dataset', gen.shape)
    save_npz('%s/chr.%s.%d.gen' % (out_directory, chrom, subfile), gen)

np.save('%s/chr.%s.gen.coordinates' % (out_directory, chrom), np.asarray(chrom_coord, dtype=int))

print('Completed in ', time.time()-t0, 'sec')

