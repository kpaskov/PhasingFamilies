import sys
import gzip
import numpy as np
import scipy.sparse as sparse
from os import listdir

data_dir = sys.argv[1]

pass_from_gen = False
if len(sys.argv)>2 and sys.argv[2] == '--pass_from_gen':
	pass_from_gen = True
	print('Generate PASS from genotypes.')

chroms = [str(x) for x in range(1, 23)] + ['X', 'Y']

for chrom in chroms:
    print(chrom, end=' ')
    
    if pass_from_gen:
        # load genotypes
        gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])

        # Pull data together
        A = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

        # AF filter
        homalt = (A==2).sum(axis=0).A.flatten()
        het = (A==1).sum(axis=0).A.flatten()
        miss = (A<0).sum(axis=0).A.flatten()
        af = (2*homalt + het)/(2*A.shape[0] - 2*miss)
        percent_miss = miss/A.shape[0]

        is_pass = (percent_miss < 0.1)

    else:
        is_pass = []
        with gzip.open('%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom), 'rt') as f:
            for line in f:
                pieces = line.strip().split('\t')
                is_pass.append(pieces[6] == 'PASS')
        is_pass = np.array(is_pass)
            
    # pull snp positions
    pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))    

    chrom_int = 23 if chrom == 'X' else 24 if chrom == 'Y' else int(chrom)
    np.save('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom), np.hstack((chrom_int*np.ones((pos_data.shape[0], 1), dtype=int), pos_data[:, 1:3], is_pass[:, np.newaxis])))

