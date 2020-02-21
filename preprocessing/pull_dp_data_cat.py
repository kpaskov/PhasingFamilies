from collections import defaultdict
import numpy as np
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
with gzip.open(vcf_file, 'rt') as f:

    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull genotypes from vcf
    m = len(line.strip().split('\t')[9:])
    n = np.load('%s/chr.%s.gen.coordinates.npy' % (out_directory, chrom)).shape[0]

    dp = np.zeros((m, n), dtype=np.int8)
    
    line = next(f)
    no_depth = 0
    for j, line in enumerate(f):
        pieces = line.split('\t', maxsplit=1)

        if pieces[0] == chrom or (chrom == 'X' and pieces[0] == '23') or (chrom == 'Y' and pieces[0] == '24') or (chrom == 'MT' and pieces[0] == '25'):

            pieces = line.strip().split('\t')
            fmt = pieces[8].strip().split(':')

            dp_index = fmt.index('DP')
            for i, piece in enumerate(pieces[9:]):
                segment = piece.split(':', maxsplit=dp_index+1)

                try:
                    dp[i, j] = min(int(segment[dp_index]), 100)
                except:
                    no_depth += 1

print('Trouble finding depth for %d calls.' % no_depth)
    
np.save('%s/chr.%s.dp' % (out_directory, chrom), dp)
print('Completed in ', time.time()-t0, 'sec')

