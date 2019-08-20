import numpy as np
import time
import gzip
import sys

t0 = time.time()

# Pull arguments
vcf_file = sys.argv[1]
pos_file = sys.argv[2]
out_directory = sys.argv[3]
chrom = sys.argv[4]

# pull positions of interest
positions_of_interest = []
with open(pos_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if pieces[0] == chrom:
            positions_of_interest.append(int(pieces[1]))
position_to_index = dict([(x, i) for i, x in enumerate(positions_of_interest)])
n = len(positions_of_interest)

# Pull data from vcf
with gzip.open(vcf_file, 'rt') as f:

    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull sample_ids and write to file
    sample_ids = line.strip().split('\t')[9:]
    
    # Pull AD from vcf
    m = len(sample_ids)
    ad = -np.ones((m, n, 2), dtype=int)

    line = next(f)
    subfile = 0
    num_lines = 0

    for j, line in enumerate(f):
        pieces = line.split('\t', maxsplit=3)
        pos = int(pieces[1])
        if pieces[0] == chrom and pos in position_to_index:
            pieces = line.split('\t')

            if len(pieces[3]) == 1 and pieces[3] != '.' and len(pieces[4]) == 1 and pieces[4] != '.':
                format = pieces[8].strip().split(':')
                ad_index = format.index('AD')

                for i, piece in enumerate(pieces[9:]):
                    segment = piece.split(':', maxsplit=ad_index+1)

                    ad1, _, ad2 = segment[ad_index].partition(',')
                    if ad2 == '' or ',' in ad2:
                        print('parse error')
                    else:
                        ad[position_to_index[pos], i, :] = int(ad1), int(ad2)
                        
    # Save to file
    np.save('%s/chr.%s.ad.ofinterest' % (out_directory, chrom), ad)

print('Completed in ', time.time()-t0, 'sec')

