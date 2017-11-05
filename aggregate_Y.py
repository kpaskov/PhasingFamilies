import os
import sys
import numpy as np

# Pull arguments
chromosome = sys.argv[1]

# Load variants
num_variants = 0
with open('data/v34.%s.txt' % chromosome, 'r') as f:
    for line in f:
        num_variants += 1

variants = np.zeros((num_variants))
with open('data/v34.%s.txt' % chromosome, 'r') as f:
    for i, line in enumerate(f):
        variants[i] = int(line.strip())

print('Num variants:', len(variants))

# Load families
family_ids = []
for filename in os.listdir('raw_data'):
    if filename.endswith('%s.npz' % chromosome):
        data = np.load('raw_data/%s' % filename)
        if 'X' in data and 'Y' in data and 'row_indices' in data and 'col_indices' in data:
            family_ids.append(filename.split('.')[0])
family_ids.sort()

print('Num families with data:', len(family_ids))
    
# Aggregate Ys
fullY = np.zeros((4*len(family_ids), num_variants), np.int8)

# Load data
for i, family_id in enumerate(family_ids):
    data = np.load('raw_data/%s.v34.%s.npz' % (family_id, chromosome))
    fullY[(4*i):(4*(i+1)), data['col_indices']] = data['Y']

np.savez_compressed('data/Y%s' % chromosome, 
            Y=fullY, family_ids=family_ids, variant_positions=variants)

print('Aggregated Ys.', fullY.shape)

