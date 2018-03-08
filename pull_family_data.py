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
    i_s, j_s, gen_v = [], [], []
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
                i_s.append(i)
                j_s.append(j)
                gen_v.append(gt)

    gen = csc_matrix((gen_v, (i_s, j_s)), shape=(m, n), dtype=np.int8)
    print('Full dataset', gen.shape)

    # Save to file
    save_npz('%s/chr.%s.gen' % (out_directory, chrom), gen)
    np.savez_compressed('%s/chr.%s.gen.anno' % (out_directory, chrom), sample_ids=pieces, m=m, n=n)

    # for family_id, family in families.items():
    #     family_rows = np.array(family.get_vcf_indices())
    #     if family_rows.shape[0] > 3:
    #         family_cols = np.where(np.logical_and(np.sum(ad1[family_rows, :], axis=0) > 0, np.sum(ad2[family_rows, :], axis=0) > 0))[0] # Remove completely homozygous entries
    #         family_gen, family_ad1, family_ad2 = gen[np.ix_(family_rows, family_cols)], ad1[np.ix_(family_rows, family_cols)], ad2[np.ix_(family_rows, family_cols)]

    #         np.savez_compressed('%s/%s_%s_%s.%s.gen.ad' % (out_directory, family_id[0], family_id[1], family_id[2], chrom),
    #                 gen=family_gen, ad1=family_ad1, ad2=family_ad2,
    #                 row_indices=family_rows, col_indices=family_cols, 
    #                 m=m, n=n, sample_ids=family.get_sample_ids())


print('Completed in ', time.time()-t0, 'sec')

