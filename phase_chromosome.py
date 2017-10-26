from collections import defaultdict
import numpy as np
import time
import gzip
from itertools import product
import sys

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

# Mappings
phase_map = {
    (0, 0, 0): (0, 0, 0, 0),
    (0, 0, 1): (-1, -1, -1, -1), # non-mendelian
    (0, 0, 2): (-1, -1, -1, -1), # non-mendelian
    (0, 1, 0): (0, 0, 0, 1),
    (0, 1, 1): (0, 0, 1, 0),
    (0, 1, 2): (-1, -1, -1, -1), # non-mendelian
    (0, 2, 0): (-1, -1, -1, -1), # non-mendelian
    (0, 2, 1): (0, 0, 1, 1),
    (0, 2, 2): (-1, -1, -1, -1), # non-mendelian
    (1, 0, 0): (0, 1, 0, 0),
    (1, 0, 1): (1, 0, 0, 0),
    (1, 0, 2): (-1, -1, -1, -1), # non-mendelian
    (1, 1, 0): (0, 1, 0, 1),
    (1, 1, 1): (-1, -1, -1, -1), # unknown phase
    (1, 1, 2): (1, 0, 1, 0),
    (1, 2, 0): (-1, -1, -1, -1), # non-mendelian
    (1, 2, 1): (0, 1, 1, 1),
    (1, 2, 2): (1, 0, 1, 1),
    (2, 0, 0): (-1, -1, -1, -1), # non-mendelian
    (2, 0, 1): (1, 1, 0, 0),
    (2, 0, 2): (-1, -1, -1, -1), # non-mendelian
    (2, 1, 0): (-1, -1, -1, -1), # non-mendelian
    (2, 1, 1): (1, 1, 0, 1),
    (2, 1, 2): (1, 1, 1, 0),
    (2, 2, 0): (-1, -1, -1, -1), # non-mendelian
    (2, 2, 1): (-1, -1, -1, -1), # non-mendelian
    (2, 2, 2): (1, 1, 1, 1),
}

# Algorithm


def rough_phase(data, child_index=2):
    return np.apply_along_axis(lambda x: phase_map[(x[0], x[1], x[child_index])], 0, data)

def phase(data, X):
    m, _, n = X.shape
    all_combinations = np.array(list(product([0, 1], repeat=4))).T
    Y = np.zeros((4, n))
    for i in range(n):
        diff = np.outer(data[:, i], np.ones(all_combinations.shape[1])) - X[:, :, i].dot(all_combinations)
        index = np.argmin(np.sum(np.abs(diff), axis=0))
        Y[:, i] = all_combinations[:, index]

    return Y

def detect_recombination(data, Y, switch_cost=50):
    m, n = data.shape
    X = np.zeros((m, 4, n))
    X[0, 0, :] = X[0, 1, :] = 1 # Mom always has m1, m2
    X[1, 2, :] = X[1, 3, :] = 1 # Dad always has p1, p2
    X[2, 0, :] = X[2, 2, :] = 1 # Child1 always has m1, p1
    
    # genotype possibilities for children
    # m1p1, m1p2, m2p1, m2p2
    Z = np.array([[1, 0, 1, 0], 
                  [1, 0, 0, 1], 
                  [0, 1, 1, 0], 
                  [0, 1, 0, 1]]).dot(Y)
    transition_costs = np.array([[0, 1, 1, 2],
                                 [1, 0, 2, 1],
                                 [1, 2, 0, 1],
                                 [2, 1, 1, 0]])*switch_cost
    index_to_indices = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    for i in range(3, m):
        # for each child
        
        # Forward sweep
        dp = [[(0, None)]*4]
        for j in range(n):
            gen = data[i, j]
            # consider cost of all 16 combinations
            possible_transitions = list(product(range(4), repeat=2))
            costs = [(dp[-1][k][0]+transition_costs[k, l]+abs(gen-Z[l, j]), k) for l, k in possible_transitions]
            dp.append([min(costs[:4]), min(costs[4:8]), min(costs[8:12]), min(costs[12:16])])
                
        #print(dp[:5])
        #print(dp[-5:])
        
        # Backward sweep
        index = n-1
        cost, prev = min(dp[index])
        X[i, index_to_indices[prev], index] = 1
        while prev is not None:
            X[i, index_to_indices[prev], index-1] = 1
            index -= 1
            _, prev = min(dp[index])
            
        #print(X[i, :, :])
    return X

def to_genotype(X, Y):
    m, _, n = X.shape
    genotype = np.zeros((m, n))
    for i in range(m):
        genotype[i, :] = np.sum(X[i, :, :]*Y, axis=0)
    return genotype

t0 = time.time()

# Pull arguments
vcf_file = sys.argv[1]
ped_file = sys.argv[2]
out_directory = sys.argv[3]

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

# Pull data from vcf
with gzip.open(vcf_file, 'rt') as f:
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull header and add vcf indices
    pieces = line.strip().split('\t')
    for i, ind_id in enumerate(pieces[9:]):
        if ind_id in member_to_family:
            families[member_to_family[ind_id]].add_vcf_index(ind_id, i)
    line = next(f)
    
    # Remove families not included in vcf
    families = dict([(i, x) for i, x in families.items() if len(x.get_vcf_indices()) > 0])
    print('Num families with genomic data:', len(families))

    have_genomic = sum([len([y for j, y in x.members.items() if y.vcf_index is not None]) for i, x in families.items()])
    print('Num individuals with genomic data', have_genomic)
    print('Num individuals missing genomic data', sum([len(x.members) for i, x in families.items()])-have_genomic)
    
    # Load genotypes into numpy arrays
    n = len(pieces)-9
    gen_mapping = {b'./.': -1, b'0/0': 0, b'0/1': 1, b'1/0': 1, b'1/1': 2}
    converter = lambda gen:gen_mapping[gen[:3]]
    vcf_indices = range(9, n+9)
    data = np.loadtxt(f, dtype=np.int8, converters=dict(zip(vcf_indices, [converter]*n)), delimiter='\t', usecols=vcf_indices).T

print('Full dataset', data.shape)

print(len([x for x, f in families.items() if len(f.parents_to_children) > 1]), 'families discarded due to complex family structure')
for family_id, family in families.items():
    if len(family.parents_to_children) == 1:
        print(family_id)

        family_data = data[family.get_vcf_indices(), :]

        # Remove completely homozygous ref entries
        family_data = family_data[:, ~(family_data==0).all(axis=0)]
        #print('Remove homozygous ref entries', family_data.shape)

        # Remove rows with missing entries
        family_data = family_data[:, (family_data!=-1).all(axis=0)]
        #print('Remove missing entries', family_data.shape)

        Y0 = rough_phase(family_data)
        X1 = detect_recombination(family_data, Y0, switch_cost=50)
        Y1 = phase(family_data, X1)
        X2 = detect_recombination(family_data, Y1, switch_cost=50)
        Y2 = phase(family_data, X2)
        #X3 = detect_recombination(family_data, Y2, switch_cost=50)
        #Y3 = phase(family_data, X3)

        np.savez_compressed(out_directory + '/' + family_id + '.' + vcf_file.split('/')[-1][:-7], X=X2, Y=Y2)

print('Completed in ', time.time()-t0, 'sec')

