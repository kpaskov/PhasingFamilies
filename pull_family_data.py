from collections import defaultdict
import numpy as np
import time
import gzip
from itertools import product
import sys

class Individual:
    def __init__(self, vcf_index):
        self.vcf_index = vcf_index


class Family:
    def __init__(self):
        self.id = None
        self.mother = None
        self.father = None
        self.children = []

    def add_trio(self, mother, father, child):
        self.mother = mother
        self.father = father
        self.children.append(child)

    def get_vcf_indices(self):
        return [self.mother.vcf_index, self.father.vcf_index] + [child.vcf_index for child in self.children]

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
    gen = np.zeros((m, n), dtype=np.int8)
    ad = np.zeros((m, n, 2), dtype=int)

    for i, line in enumerate(f):
        pieces = line.split('\t')
        format = pieces[8].strip().split(':')
        gen_index = format.index('GT')
        ad_index = format.index('AD')
        for j, piece in enumerate(pieces[9:]):
            segment = piece.split(':', maxsplit=ad_index+1)

            gen[j, i] = gen_mapping[segment[gen_index]]

            ad_segment = segment[ad_index].split(',')
            if len(ad_segment) == 2:
                ad[j, i, :] = [int(a) for a in segment[ad_index].split(',')]  

    return gen, ad

t0 = time.time()

# Pull arguments
vcf_file = sys.argv[1]
ped_file = sys.argv[2]
out_directory = sys.argv[3]
chrom = sys.argv[4]

# Pull data from vcf
n = variants_in_vcf(vcf_file)
families = defaultdict(Family)

with gzip.open(vcf_file, 'rt') as f, open(ped_file, 'r') as pedf:

    # Pull header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull header and create individuals
    pieces = line.strip().split('\t')[9:]
    individuals = dict([(ind_id, Individual(i)) for i, ind_id in enumerate(pieces)])

    # Create families
    for line in pedf:
        fam_id, child_id, father_id, mother_id, the_rest = line.split('\t', maxsplit=4)
        if father_id in individuals and mother_id in individuals and child_id in individuals:
             families[(fam_id, mother_id, father_id)].add_trio(individuals[mother_id], individuals[father_id], individuals[child_id])

    print('Num families with genomic data:', len(families))
    print('Num individuals with genomic data', len(individuals))

    # Load genotypes into numpy arrays
    m = len(pieces)
    line = next(f)
    gen, ad = read_vcf(f, m, n)
    print('Full dataset', gen.shape)

for family_id, family in families.items():
    family_rows = np.array(family.get_vcf_indices())
    family_cols = np.where(np.logical_and(np.sum(ad[family_rows, :, 0], axis=0) > 0, np.sum(ad[family_rows, :, 1], axis=0) > 0))[0] # Remove completely homozygous entries
    family_gen, family_ad = gen[np.ix_(family_rows, family_cols)], ad[np.ix_(family_rows, family_cols, [0, 1])]

    np.savez_compressed('%s/%s_%s_%s.%s.gen.ad' % (out_directory, family_id[0], family_id[1], family_id[2], chrom),
            gen=family_gen, ad=family_ad, 
            row_indices=family_rows, col_indices=family_cols, 
            m=m, n=n)
print('Completed in ', time.time()-t0, 'sec')

