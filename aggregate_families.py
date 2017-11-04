import os
import sys
import numpy as np

# Pull arguments
chromosome = sys.argv[1]

# Load data
family_ids, Xs, Ys, row_indices, col_indices = [], [], [], [], []
for filename in os.listdir('data'):
    if filename.endswith('%s.npz' % chromosome):
        data = np.load('data/%s' % filename)
        if 'X' in data and 'Y' in data and 'row_indices' in data and 'col_indices' in data:
            family_ids.append(filename.split('.')[0])
            Xs.append(data['X'])
            Ys.append(data['Y'])
            row_indices.append(data['row_indices'])
            col_indices.append(data['col_indices'])

print('Num families with data:', len(family_ids))

# Load variants
variants = []
with open('data/v34.%s.txt' % chromosome, 'r') as f:
    for line in f:
        variants.append(int(line.strip()))

# Aggregate recombinations
maternal_recombinations = []
paternal_recombinations = []
for k, X in enumerate(Xs):
    if X is not None:
        m, _, n = X.shape
        for j in range(3, m):
            maternal_recombinations.extend([col_indices[k][i] for i in range(n-1) if X[j, 0, i] != X[j, 0, i+1]])
            paternal_recombinations.extend([col_indices[k][i] for i in range(n-1) if X[j, 2, i] != X[j, 2, i+1]])
   
maternal_recombinations.sort()
paternal_recombinations.sort()
print(len(maternal_recombinations), len(paternal_recombinations))

with open('data/recomb%s.txt' % chromosome, 'w+') as f:
    for mr in maternal_recombinations:
        f.write("%d\tM\n" % mr)
    for pr in paternal_recombinations:
        f.write("%d\tP\n" % pr)

print('Aggregated recombinations.')
print('Num maternal:', len(maternal_recombinations), 'Num paternal', len(paternal_recombinations))


# Aggregate Ys
fullY = np.zeros((4*len(families), len(variants)), np.int8)

for i, Y in enumerate(Ys):
    fullY[(4*i):(4*(i+1)), col_indices[i]] = Y

np.savez_compressed('data/Y%s.txt' % chromosome, 
            Y=fullY, family_ids=family_ids, variant_positions=variants)

print('Aggregated Ys.', fullY.shape)

