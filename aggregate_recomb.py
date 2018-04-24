import os
import sys
import numpy as np
import gzip

# Pull arguments
chromosome = sys.argv[1]
ped_file = sys.argv[2]
vcf_file = sys.argv[3]

# Pull asd status from ped file
id_to_asd = {}
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        fam_id, child_id, father_id, mother_id = pieces[0:4]
        has_asd = (pieces[5]=='2')
        
        id_to_asd[child_id] = has_asd
print(len(id_to_asd), list(id_to_asd.items())[:10])

# Map IDs to indices
index_to_asd = {}
with gzip.open(vcf_file, 'rb') as f:
    line = next(f).decode()
    # skip header
    while line.startswith('##'):
        line = next(f).decode()

    for i, ind_id in enumerate(line.strip().split('\t')[9:]):
        if ind_id in id_to_asd:
            index_to_asd[i] = id_to_asd[ind_id]
print(len(index_to_asd), list(index_to_asd.items())[:10])

# Load data
maternal_recombinations, paternal_recombinations = [], []
asd_maternal_recombinations, asd_paternal_recombinations = [], []
control_maternal_recombinations, control_paternal_recombinations = [], []

for filename in os.listdir('raw_data'):
    if filename.endswith('v34.%s.npz' % chromosome):
        data = np.load('raw_data/%s' % filename)
        if 'X' in data and 'Y' in data and 'row_indices' in data and 'col_indices' in data:
            X = data['X']
            row_indices = data['row_indices']
            col_indices = data['col_indices']
            m, _, n = X.shape
            diff = X[:, :, :-1] - X[:, :, 1:]

            mat_rec_indices = np.where(diff[:, 0, :]!=0)
            pat_rec_indices = np.where(diff[:, 2, :]!=0)

            maternal_recombinations.extend([col_indices[i] for i in mat_rec_indices[1]])
            paternal_recombinations.extend([col_indices[i] for i in pat_rec_indices[1]])

            asd_maternal_recombinations.extend([col_indices[j] for i, j in zip(mat_rec_indices[0], mat_rec_indices[1]) if row_indices[i] in index_to_asd and index_to_asd[row_indices[i]]])
            asd_paternal_recombinations.extend([col_indices[j] for i, j in zip(pat_rec_indices[0], pat_rec_indices[1]) if row_indices[i] in index_to_asd and index_to_asd[row_indices[i]]])

            control_maternal_recombinations.extend([col_indices[j] for i, j in zip(mat_rec_indices[0], mat_rec_indices[1]) if row_indices[i] in index_to_asd and not index_to_asd[row_indices[i]]])
            control_paternal_recombinations.extend([col_indices[j] for i, j in zip(pat_rec_indices[0], pat_rec_indices[1]) if row_indices[i] in index_to_asd and not index_to_asd[row_indices[i]]])

maternal_recombinations.sort()
paternal_recombinations.sort()
asd_maternal_recombinations.sort()
asd_paternal_recombinations.sort()
control_maternal_recombinations.sort()
control_paternal_recombinations.sort()
print('Maternal R', len(maternal_recombinations), 'Paternal R', len(paternal_recombinations),
    'ASD Maternal R', len(asd_maternal_recombinations), 'ASD Paternal R', len(asd_paternal_recombinations),
    'Control Maternal R', len(control_maternal_recombinations), 'Control Paternal R', len(control_paternal_recombinations))

with open('data/recomb%s.txt' % chromosome, 'w+') as f:
    for mr in maternal_recombinations:
        f.write("%d\tM\n" % mr)
    for pr in paternal_recombinations:
        f.write("%d\tP\n" % pr)

with open('data/asd_recomb%s.txt' % chromosome, 'w+') as f:
    for mr in asd_maternal_recombinations:
        f.write("%d\tM\n" % mr)
    for pr in asd_paternal_recombinations:
        f.write("%d\tP\n" % pr)

with open('data/control_recomb%s.txt' % chromosome, 'w+') as f:
    for mr in control_maternal_recombinations:
        f.write("%d\tM\n" % mr)
    for pr in control_paternal_recombinations:
        f.write("%d\tP\n" % pr)

print('Aggregated recombinations.')
