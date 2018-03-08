from collections import defaultdict
import numpy as np
from scipy.sparse import csc_matrix
import time
import gzip
from itertools import product
import sys

class Individual:
    def __init__(self, ind_id, vcf_index):
        self.id = ind_id
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

    def get_sample_ids(self):
        return [self.mother.id, self.father.id] + [child.id for child in self.children]

    def size(self):
        return len(self.children)+2

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
    i_s, j_s = [], []
    gen_v, ad1_v, ad2_v = [], [], []

    for j, line in enumerate(f):
        pieces = line.split('\t')
        format = pieces[8].strip().split(':')
        gen_index = format.index('GT')
        ad_index = format.index('AD')
        for i, piece in enumerate(pieces[9:]):
            segment = piece.split(':', maxsplit=ad_index+1)

            ad_segment = segment[ad_index].split(',')
            if len(ad_segment) == 2:
                i_s.append(i)
                j_s.append(j)
                gen_v.append(gen_mapping[segment[gen_index]])
                ad1_v.append(min(int(ad_segment[0]), 255))
                ad2_v.append(min(int(ad_segment[1]), 255)) 

    gen = csc_matrix((gen_v, (i_s, j_s)), shape=(m, n))
    ad1 = csc_matrix((ad1_v, (i_s, j_s)), shape=(m, n))
    ad2 = csc_matrix((ad2_v, (i_s, j_s)), shape=(m, n))
    return gen, ad1, ad2

t0 = time.time()

# Pull arguments
vcf_file = sys.argv[1]
ped_file = sys.argv[2]
out_directory = sys.argv[3]
chrom = sys.argv[4]

# Pull data from vcf
n = variants_in_vcf(vcf_file)
families = defaultdict(Family)

t1 = time.time()

with gzip.open(vcf_file, 'rt') as f, open(ped_file, 'r') as pedf:

    # Pull header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    # Pull header and create individuals
    pieces = line.strip().split('\t')[9:]
    individuals = dict([(ind_id, Individual(ind_id, i)) for i, ind_id in enumerate(pieces)])

    # Create families
    for line in pedf:
        fam_id, child_id, father_id, mother_id, the_rest = line.split('\t', maxsplit=4)
        if father_id in individuals and mother_id in individuals and child_id in individuals:
             families[(fam_id, mother_id, father_id)].add_trio(individuals[mother_id], individuals[father_id], individuals[child_id])

    family_ids = sorted(list(families.keys()))

    print('Num families with genomic data:', len(family_ids))
    print('Num individuals with genomic data', len(individuals))

    # Load genotypes into numpy arrays
    m = len(pieces)
    line = next(f)
    gen, ad1, ad2 = read_vcf(f, m, n)
    print('Full dataset', gen.shape)

    np.savez_compressed('%s/%s.gen.ad' % (out_directory, chrom),
        gen=gen, ad1=ad1, ad2=ad2, sample_ids=pieces, m=m, n=n)

    for family_id, family in families.items():
        family_rows = np.array(family.get_vcf_indices())
        family_cols = np.where(np.logical_and(np.sum(ad1[family_rows, :], axis=0) > 0, np.sum(ad2[family_rows, :], axis=0) > 0))[0] # Remove completely homozygous entries
        family_gen, family_ad1, family_ad2 = gen[np.ix_(family_rows, family_cols)], ad1[np.ix_(family_rows, family_cols)], ad2[np.ix_(family_rows, family_cols)]

        np.savez_compressed('%s/%s_%s_%s.%s.gen.ad' % (out_directory, family_id[0], family_id[1], family_id[2], chrom),
                gen=family_gen, ad1=family_ad1, ad2=family_ad2,
                row_indices=family_rows, col_indices=family_cols, 
                m=m, n=n, sample_ids=family.get_sample_ids())


print('Completed in ', time.time()-t0, 'sec')

