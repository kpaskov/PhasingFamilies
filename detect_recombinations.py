import os.path
import sys

# Pull arguments
ped_file = sys.argv[1]
data_dir = sys.argv[2]
chromosome = sys.argv[3]
out_filename = sys.argv[4]

import numpy as np

from collections import defaultdict

# Classes
class Individual:
    def __init__(self, ind_id):
        self.id = ind_id
        self.vcf_index = None
        self.autism_status = None
        self.sex = None

    def __repr__(self):
        return '%s' % self.id

class Family:

    def __init__(self):
        self.parents_to_children = defaultdict(list) # (mother_id, father_id) -> [child1, child2, ...]
        self.members = {}

    def add_trio(self, child_id, mother_id, father_id, child_autism_status, child_sex):
        if child_id not in self.members:
            self.members[child_id] = Individual(child_id)
            self.members[child_id].autism_status = child_autism_status
            self.members[child_id].sex = child_sex
        child = self.members[child_id]
        if mother_id not in self.members:
            self.members[mother_id] = Individual(mother_id)
        mother = self.members[mother_id]
        if father_id not in self.members:
            self.members[father_id] = Individual(father_id)
        father = self.members[father_id]

        self.parents_to_children[(mother_id, father_id)].append(child_id)

    def add_vcf_index(self, ind_id, vcf_index):
        if ind_id in self.members:
            self.members[ind_id].vcf_index = vcf_index
            
    def get_ordered_member_ids(self):
        ordered_member_ids = []
        for (mother_id, father_id), child_ids in self.parents_to_children.items():
            ordered_member_ids.extend([mother_id, father_id] + child_ids)
        ordered_member_ids = [x for x in ordered_member_ids if self.members[x].vcf_index is not None]
        return ordered_member_ids

    def get_info(self):
        ordered_member_ids = self.get_ordered_member_ids()
        return [(self.members[ind_id].autism_status, self.members[ind_id].sex) for ind_id in ordered_member_ids]
    
    def get_vcf_indices(self):
        ordered_member_ids = self.get_ordered_member_ids()
        return [self.members[ind_id].vcf_index for ind_id in ordered_member_ids if self.members[ind_id].vcf_index is not None]

# Pull family structure from ped file
families = {}
member_to_family = {}
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        fam_id, child_id, father_id, mother_id = pieces[0:4]
        if fam_id not in families:
            families[fam_id] = Family()
        families[fam_id].add_trio(child_id, mother_id, father_id, 
                                  'Autism' if pieces[5]=='2' else 'Control', 
                                  'Male' if pieces[4]=='1' else 'Female')
        member_to_family[child_id] = fam_id
        member_to_family[mother_id] = fam_id
        member_to_family[father_id] = fam_id
print('Num families:', len(families))

# Refine families based on data
family_ids = [k for k, v in families.items() if os.path.isfile('data/%s.v34.%s.npz' % (k, chromosome))]
print('Num families with data:', len(family_ids))

# Load data
Xs = [None]*len(family_ids)
Ys = [None]*len(family_ids)
row_indices = [None]*len(family_ids)
col_indices = [None]*len(family_ids)
for i, family_id in enumerate(family_ids):
    data = np.load('%s/%s.v34.%s.npz' % (data_dir, family_id, chromosome))
    Xs[i] = data['X']
    Ys[i] = data['Y']
    row_indices[i] = data['row_indices']
    col_indices[i] = data['col_indices']

# Pull recombinations
maternal_recombinations = []
paternal_recombinations = []
for k, X in enumerate(Xs):
    if X is not None:
        m, _, n = X.shape
        for j in range(3, m):
            maternal_recombinations.extend([col_indices[k][i] for i in range(n-1) if X[j, 0, i] != X[j, 0, i+1]])
            paternal_recombinations.extend([col_indices[k][i] for i in range(n-1) if X[j, 2, i] != X[j, 2, i+1]])
            #print(maternal_recombinations, paternal_recombinations)
   

maternal_recombinations.sort()
paternal_recombinations.sort()
print(len(maternal_recombinations), len(paternal_recombinations))

# Write to file
with open(out_filename, 'w+') as f:
    for mr in maternal_recombinations:
        f.write("%d\tM\n" % mr)
    for pr in paternal_recombinations:
        f.write("%d\tP\n" % pr)

