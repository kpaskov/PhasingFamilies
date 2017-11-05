import os
import sys
import numpy as np

# Pull arguments
chromosome = sys.argv[1]

# Load data
family_ids, Ys, col_indices = [], [], []
for filename in os.listdir('raw_data'):
    if filename.endswith('%s.npz' % chromosome):
        data = np.load('raw_data/%s' % filename)
        if 'X' in data and 'Y' in data and 'row_indices' in data and 'col_indices' in data:
            family_ids.append(filename.split('.')[0])
            #Xs.append(data['X'])
            Ys.append(data['Y'])
            #row_indices.append(data['row_indices'])
            col_indices.append(data['col_indices'])

print('Num families with data:', len(family_ids))

# Load variants
variants = []
with open('data/v34.%s.txt' % chromosome, 'r') as f:
    for line in f:
        variants.append(int(line.strip()))

# Aggregate Ys
fullY = np.zeros((4*len(family_ids), len(variants)), np.int8)

for i, Y in enumerate(Ys):
    fullY[(4*i):(4*(i+1)), col_indices[i]] = Y

np.savez_compressed('data/Y%s.txt' % chromosome, 
            Y=fullY, family_ids=family_ids, variant_positions=variants)

print('Aggregated Ys.', fullY.shape)

