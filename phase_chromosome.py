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

# Custom IO for our large vcf files
# We assume that we've iterated past the header of the vcf file
def variants_in_vcf(filename):
    n = 0
    with gzip.open(vcf_file, 'rt') as f:
        line = next(f)
        while line.startswith('#'):
            line = next(f)

        for line in f:
            n += 1
    return n


def read_vcf(f, m, n):
    # something like
    # data = np.loadtxt(f, dtype=np.int8, converters=dict(zip(vcf_indices, [converter]*n)), delimiter='\t', usecols=vcf_indices).T
    
    gen_mapping = {'./.': -1, '0/0': 0, '0/1': 1, '1/0': 1, '1/1': 2}

    # Pre-allocate memory
    data = np.zeros((m, n), dtype=np.int8)

    for i, line in enumerate(f):
        pieces = line.split('\t')[9:]
        data[:, i] = [gen_mapping[gen[:3]] for gen in pieces]

    return data

# Algorithm
def rough_phase(data, child_index=2, phase_map=phase_map):
    return np.apply_along_axis(lambda x: phase_map[(x[0], x[1], x[child_index])], 0, data)


def phase(data, X, Y):
    m, _, n = X.shape
    all_combinations = np.array(list(product([0, 1], repeat=4))).T
    for i in range(n):
        diff = np.repeat(data[:, i].reshape((m, 1)), 16, axis=1) - X[:, :, i].dot(all_combinations)
        index = np.argmin(np.sum(np.abs(diff), axis=0))
        Y[:, i] = all_combinations[:, index]


def detect_recombination(data, X, Y, switch_cost=50):
    m, n = data.shape

    X[0, 0, :] = X[0, 1, :] = 1 # Mom always has m1, m2
    X[1, 2, :] = X[1, 3, :] = 1 # Dad always has p1, p2
    X[2, 0, :] = X[2, 2, :] = 1 # Child1 always has m1, p1

    # genotype possibilities for children
    # m1p1, m1p2, m2p1, m2p2
    Z = np.array([1, 0, 1, 0, 
                  1, 0, 0, 1, 
                  0, 1, 1, 0, 
                  0, 1, 0, 1]).reshape(4, 4).dot(Y)
    transition_costs = np.array([0, 1, 1, 2,
                                 1, 0, 2, 1,
                                 1, 2, 0, 1,
                                 2, 1, 1, 0]).reshape(4, 4)*switch_cost
    index_to_indices = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    for i in range(3, m):
        # for each child
        
        # Forward sweep
        dp_cost = np.zeros((4, n+1), dtype=np.int)
        dp_arrow = np.zeros((4, n+1), dtype=np.int8)
        dp_arrow[:, 0] = -1

        
        for j in range(n):  
            # costs is a 4x4 matrix representing the cost of transitioning from i -> j         
            costs = np.repeat(dp_cost[:, j].reshape((4, 1)), 4, axis=1) # cost of i
            costs += transition_costs # transition cost
            costs += np.abs(data[i, j] - np.repeat(Z[:, j].reshape((1, 4)), 4, axis=0)) # cost of j

            dp_arrow[:, j+1] = np.argmin(costs, axis=0)
            dp_cost[:, j+1] = np.min(costs, axis=0)
            
        # Backward sweep
        index = n
        k = np.argmin(dp_cost[:, index])
        while index > 0:
            X[i, index_to_indices[k], index-1] = 1
            k = dp_arrow[k, index]
            index -= 1

def to_genotype(X, Y):
    m, _, n = X.shape
    genotype = np.zeros((m, n), dtype=np.int8)
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
n = variants_in_vcf(vcf_file)
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
    m = len(pieces)-9
    data = read_vcf(f, m, n)

print('Full dataset', data.shape)

print(len([x for x, f in families.items() if len(f.parents_to_children) > 1 or len(list(f.parents_to_children.items())[0][1]) <= 1]), 'families discarded due to complex family structure')
for family_id, family in families.items():
    if len(family.parents_to_children) == 1 and len(list(family.parents_to_children.items())[0][1]) > 1:
        print(family_id)

        family_rows = family.get_vcf_indices()
        family_data = data[family_rows, :]

        family_cols = np.logical_and(~(family_data==0).all(axis=0), # Remove completely homozygous ref entries
                                    (family_data!=-1).all(axis=0)) # Remove rows with missing entries
        family_data = family_data[:, family_cols]

        prev_time = time.time()
        Y = rough_phase(family_data)
        print('Rough Phase', time.time()-prev_time, 'sec')

        m, n = family_data.shape
        X = np.zeros((m, 4, n))

        prev_time = time.time()
        detect_recombination(family_data, X, Y, switch_cost=50)
        print('Detect Recomb', time.time()-prev_time, 'sec')

        prev_time = time.time()
        phase(family_data, X, Y)
        print('Phase', time.time()-prev_time, 'sec')

        family_rows = np.array(family_rows)
        family_cols = np.where(family_cols)[0]
        np.savez_compressed(out_directory + '/' + family_id + '.' + vcf_file.split('/')[-1][:-7], 
            X=X, Y=Y, data=family_data, row_indices=family_rows, col_indices=family_cols)
print('Completed in ', time.time()-t0, 'sec')

