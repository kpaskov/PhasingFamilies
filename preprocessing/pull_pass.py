import sys
import gzip
import numpy as np

data_dir = sys.argv[1]
all_pass = False
if len(sys.argv)>2 and sys.argv[2] == 'all_pass=True':
	all_pass = True
	print('All pass.')

chroms = [str(x) for x in range(1, 23)]

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

    pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))
    np.save('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom), np.hstack((pos_data, np.array(passes)[:, np.newaxis])))
                
                