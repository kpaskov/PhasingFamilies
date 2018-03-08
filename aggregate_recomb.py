import os
import sys
import numpy as np

# Pull arguments
chromosome = sys.argv[1]

# Load data
maternal_recombinations = []
paternal_recombinations = []
for filename in os.listdir('raw_data'):
    if filename.endswith('v34.%s.npz' % chromosome):
        data = np.load('raw_data/%s' % filename)
        if 'X' in data and 'Y' in data and 'row_indices' in data and 'col_indices' in data:
            X = data['X']
            col_indices = data['col_indices']
            m, _, n = X.shape
            diff = X[:, :, :-1] - X[:, :, 1:]

            maternal_recombinations.extend([col_indices[i] for i in np.where(diff[:, 0, :]!=0)[1]])
            paternal_recombinations.extend([col_indices[i] for i in np.where(diff[:, 2, :]!=0)[1]])

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