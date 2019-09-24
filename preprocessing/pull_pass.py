import sys
import gzip
import numpy as np

data_dir = sys.argv[1]
all_pass = False
if len(sys.argv)>2 and sys.argv[2] == 'all_pass=True':
	all_pass = True
	print('All pass.')

chroms = [str(x) for x in range(1, 23)] + ['X', 'Y']

for chrom in chroms:
    print(chrom, end=' ')
    passes = []
    with gzip.open('%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom), 'rt') as f:
        for line in f:
            pieces = line.strip().split('\t')
            if all_pass:
            	passes.append(True)
            else:
            	passes.append(pieces[6] == 'PASS')
            
    # pull snp positions
    pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))

    chrom_int = 23 if chrom == 'X' else 24 if chrom == 'Y' else int(chrom)
    np.save('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom), np.hstack((chrom_int*np.ones((pos_data.shape[0], 1), dtype=int), pos_data[:, 1:3], np.asarray(passes)[:, np.newaxis])))