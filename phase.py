import sys
import time
import gzip
import numpy as np

class Individual:
	def __init__(self, ind_id):
		self.id = ind_id
		self.vcf_index = None

	def __repr__(self):
		return '%s' % self.id

class Family:
	
	def __init__(self):
		self.id_to_index = dict()
		self.child_to_parents = dict() # child_id -> (mother_id, father_id)
		self.members = []

	def find_or_add_individual(self, ind_id):
		if ind_id in self.id_to_index:
			return self.members[self.id_to_index[ind_id]]
		else:
			ind = Individual(ind_id)
			self.id_to_index[ind_id] = len(self.members)
			self.members.append(ind)
			return ind
	
	def add_trio(self, child_id, mother_id, father_id):
		child = self.find_or_add_individual(child_id)
		mother = self.find_or_add_individual(mother_id)
		father = self.find_or_add_individual(father_id)

		self.child_to_parents[child_id] = (mother_id, father_id)

	def add_vcf_index(self, ind_id, vcf_index):
		if ind_id in self.id_to_index:
			self.members[self.id_to_index[ind_id]].vcf_index = vcf_index

	def order(self):
		trio = list(self.child_to_parents.items())[0]
		order = [self.id_to_index[trio[1][0]], self.id_to_index[trio[1][1]], self.id_to_index[trio[0]]] 
		order = order + [self.id_to_index[m.id] for m in self.members if self.id_to_index[m.id] not in order]
		return order


# A series of mappings from mom, dad, child genotypes to variant type.
mendelian_inheritance_map = {
	(-2, -2): {-2},
	(-2, 0): {-2, 0},
	(-2, 2): {0},
	(0, -2): {-2, 0},
	(0, 0): {-2, 0, 2},
	(0, 2): {0, 2},
	(2, -2): {0},
	(2, 0): {0, 2},
	(2, 2): {2}
}

phase_map = {
	(-2, -2, -2): (-1, -1, -1, -1),
	(-2, 0, -2): (-1, -1, -1, 1),
	(-2, 0, 0): (-1, -1, 1, -1),
	(-2, 2, 0): (-1, -1, 1, 1),
	(0, -2, -2): (-1, 1, -1, -1),
	(0, -2, 0): (1, -1, -1, -1),
	(0, 0, -2): (-1, 1, -1, 1),
	(0, 0, 0): (0, 0, 0, 0),
	(0, 0, 2): (1, -1, 1, -1),
	(0, 2, 0): (-1, 1, 1, 1),
	(0, 2, 2): (1, -1, 1, 1),
	(2, -2, 0): (1, 1, -1, -1),
	(2, 0, 0): (1, 1, -1, 1),
	(2, 0, 2): (1, 1, 1, -1),
	(2, 2, 2): (1, 1, 1, 1)
}


# (mom, dad, child) -> (m1, m2, d1, d2)
# where the child has inherited the m1 and d1 chromosomes
genotype_phase_map = {
	(-9, -9, -9): (0, 0, 0, 0),
	(-9, -9, -2): (-1, 0, -1, 0),
	(-9, -9, 0): (0, 0, 0, 0),
	(-9, -9, 2): (1, 0, 1, 0),
	(-9, -2, -9): (0, 0, -1, -1),
	(-9, -2, -2): (-1, 0, -1, -1),
	(-9, -2, 0): (1, 0, -1, -1),
	(-9, -2, 2): (0, 0, 0, 0), # non-mendelian
	(-9, 0, -9): (0, 0, 0, 0),
	(-9, 0, -2): (-1, 0, -1, 1),
	(-9, 0, 0): (0, 0, 0, 0),
	(-9, 0, 2): (1, 0, 1, -1),
	(-9, 2, -9): (0, 0, 1, 1),
	(-9, 2, -2): (0, 0, 0, 0), # non-mendelian
	(-9, 2, 0): (-1, 0, 1, 1),
	(-9, 2, 2): (1, 0, 1, 1),

	(-2, -9, -9): (-1, -1, 0, 0),
	(-2, -9, -2): (-1, -1, -1, 0),
	(-2, -9, 0): (-1, -1, 1, 0),
	(-2, -9, 2): (0, 0, 0, 0), # non-mendelian
	(-2, -2, -9): (-1, -1, -1, -1),
	(-2, -2, -2): (-1, -1, -1, -1),
	(-2, -2, 0): (0, 0, 0, 0), # non-mendelian
	(-2, -2, 2): (0, 0, 0, 0), # non-mendelian
	(-2, 0, -9): (-1, -1, 0, 0),
	(-2, 0, -2): (-1, -1, -1, 1),
	(-2, 0, 0): (-1, -1, 1, -1),
	(-2, 0, 2): (0, 0, 0, 0), # non-mendelian
	(-2, 2, -9): (-1, -1, 1, 1),
	(-2, 2, -2): (0, 0, 0, 0), # non-mendelian
	(-2, 2, 0): (-1, -1, 1, 1),
	(-2, 2, 2): (0, 0, 0, 0), # non-mendelian

	(0, -9, -9): (0, 0, 0, 0),
	(0, -9, -2): (-1, 1, -1, 0),
	(0, -9, 0): (0, 0, 0, 0),
	(0, -9, 2): (1, -1, 1, 0),
	(0, -2, -9): (0, 0, -1, -1),
	(0, -2, -2): (-1, 1, -1, -1),
	(0, -2, 0): (1, -1, -1, -1),
	(0, -2, 2): (0, 0, 0, 0), # non-mendelian
	(0, 0, -9): (0, 0, 0, 0),
	(0, 0, -2): (-1, 1, -1, 1),
	(0, 0, 0): (0, 0, 0, 0),
	(0, 0, 2): (1, -1, 1, -1),
	(0, 2, -9): (0, 0, 1, 1),
	(0, 2, -2): (0, 0, 0, 0), # non-mendelian
	(0, 2, 0): (-1, 1, 1, 1),
	(0, 2, 2): (1, -1, 1, 1),

	(2, -9, -9): (1, 1, 0, 0),
	(2, -9, -2): (0, 0, 0, 0), # non-mendelian
	(2, -9, 0): (1, 1, -1, 0),
	(2, -9, 2): (1, 1, 1, 0),
	(2, -2, -9): (1, 1, -1, -1),
	(2, -2, -2): (0, 0, 0, 0), # non-mendelian
	(2, -2, 0): (1, 1, -1, -1),
	(2, -2, 2): (0, 0, 0, 0), # non-mendelian
	(2, 0, -9): (1, 1, 0, 0),
	(2, 0, -2): (0, 0, 0, 0), # non-mendelian
	(2, 0, 0): (1, 1, -1, 1),
	(2, 0, 2): (1, 1, 1, -1),
	(2, 2, -9): (1, 1, 1, 1),
	(2, 2, -2): (0, 0, 0, 0), # non-mendelian
	(2, 2, 0): (0, 0, 0, 0), # non-mendelian
	(2, 2, 2): (1, 1, 1, 1),
}

# (mom, dad, male child) -> (m1, m2, d)
# where the male child has inherited the m1 X chromosome
x_male_genotype_phase_map = {
	(-9, -9, -9): (0, 0, 0),
	(-9, -9, -2): (-1, 0, 0),
	(-9, -9, 2): (1, 0, 0),
	(-9, -2, -9): (0, 0, -1),
	(-9, -2, -2): (-1, 0, -1),
	(-9, -2, 2): (1, 0, -1),
	(-9, 2, -9): (0, 0, 1),
	(-9, 2, -2): (-1, 0, 1),
	(-9, 2, 2): (1, 0, 1),

	(-2, -9, -9): (-1, -1, 0),
	(-2, -9, -2): (-1, -1, 0),
	(-2, -9, 2): (0, 0, 0), # non-mendelian
	(-2, -2, -9): (-1, -1, -1),
	(-2, -2, -2): (-1, -1, -1),
	(-2, -2, 2): (0, 0, -1), # non-mendelian
	(-2, 2, -9): (-1, -1, 1),
	(-2, 2, -2): (-1, -1, 1),
	(-2, 2, 2): (0, 0, 1), # non-mendelian

	(0, -9, -9): (0, 0, 0),
	(0, -9, -2): (-1, 1, 0),
	(0, -9, 2): (1, -1, 0),
	(0, -2, -9): (0, 0, -1),
	(0, -2, -2): (-1, 1, -1),
	(0, -2, 2): (1, -1, -1),
	(0, 2, -9): (0, 0, 1),
	(0, 2, -2): (-1, 1, 1),
	(0, 2, 2): (1, -1, 1),

	(2, -9, -9): (1, 1, 0),
	(2, -9, -2): (0, 0, 0), # non-mendelian
	(2, -9, 2): (1, 1, 0),
	(2, -2, -9): (1, 1, -1),
	(2, -2, -2): (0, 0, -1), # non-mendelian
	(2, -2, 2): (1, 1, -1),
	(2, 2, -9): (1, 1, 1),
	(2, 2, -2): (0, 0, 1), # non-mendelian
	(2, 2, 2): (1, 1, 1),
}

# (mom, dad, female child) -> (m1, m2, d)
# where the female child has inherited the m1 and d X chromosomes
x_female_genotype_phase_map = {
	(-9, -9, -9): (0, 0, 0),
	(-9, -9, -2): (-1, 0, -1),
	(-9, -9, 0): (0, 0, 0),
	(-9, -9, 2): (1, 0, 1),
	(-9, -2, -9): (0, 0, -1),
	(-9, -2, -2): (-1, 0, -1),
	(-9, -2, 0): (1, 0, -1),
	(-9, -2, 2): (0, 0, 0), # non-mendelian
	(-9, 2, -9): (0, 0, 1),
	(-9, 2, -2): (0, 0, 0), # non-mendelian
	(-9, 2, 0): (-1, 0, 1),
	(-9, 2, 2): (1, 0, 1),

	(-2, -9, -9): (-1, -1, 0),
	(-2, -9, -2): (-1, -1, -1),
	(-2, -9, 0): (-1, -1, 1),
	(-2, -9, 2): (0, 0, 0), # non-mendelian
	(-2, -2, -9): (-1, -1, -1),
	(-2, -2, -2): (-1, -1, -1),
	(-2, -2, 0): (0, 0, 0), # non-mendelian
	(-2, -2, 2): (0, 0, 0), # non-mendelian
	(-2, 2, -9): (-1, -1, 1),
	(-2, 2, -2): (0, 0, 0), # non-mendelian
	(-2, 2, 0): (-1, -1, 1),
	(-2, 2, 2): (0, 0, 0), # non-mendelian

	(0, -9, -9): (0, 0, 0),
	(0, -9, -2): (-1, 1, -1),
	(0, -9, 0): (0, 0, 0),
	(0, -9, 2): (1, -1, 1),
	(0, -2, -9): (0, 0, -1),
	(0, -2, -2): (-1, 1, -1),
	(0, -2, 0): (1, -1, -1),
	(0, -2, 2): (0, 0, 0), # non-mendelian
	(0, 2, -9): (0, 0, 1),
	(0, 2, -2): (0, 0, 0), # non-mendelian
	(0, 2, 0): (-1, 1, 1),
	(0, 2, 2): (1, -1, 1),

	(2, -9, -9): (1, 1, 0),
	(2, -9, -2): (0, 0, 0), # non-mendelian
	(2, -9, 0): (1, 1, -1),
	(2, -9, 2): (1, 1, 1),
	(2, -2, -9): (1, 1, -1),
	(2, -2, -2): (0, 0, 0), # non-mendelian
	(2, -2, 0): (1, 1, -1),
	(2, -2, 2): (0, 0, 0), # non-mendelian
	(2, 2, -9): (1, 1, 1),
	(2, 2, -2): (0, 0, 0), # non-mendelian
	(2, 2, 0): (0, 0, 0), # non-mendelian
	(2, 2, 2): (1, 1, 1),
}

# Check genotype phase maps
for k, v in genotype_phase_map.items():
	if k[0] != -9 and (v[0] + v[1] != k[0]) and v != (0, 0, 0, 0):
		print("Mom's genotype doesn't match.", k, v)
	if k[1] != -9 and (v[2] + v[3] != k[1]) and v != (0, 0, 0, 0):
		print("Dad's genotype doesn't match.", k, v)
	if k[2] != -9 and (v[0] + v[2] != k[2]) and v != (0, 0, 0, 0):
		print("Child's genotype doesn't match.", k, v)

for k, v in x_male_genotype_phase_map.items():
	if k[0] != -9 and (v[0] + v[1] != k[0]) and v[:2] != (0, 0):
		print("Mom's genotype doesn't match (male X).", k, v)
	if k[1] != -9 and (2*v[2] != k[1]):
		print("Dad's genotype doesn't match (male X).", k, v)
	if k[2] != -9 and (2*v[0] != k[2]) and v[:2] != (0, 0):
		print("Child's genotype doesn't match (male X).", k, v)

for k, v in x_female_genotype_phase_map.items():
	if k[0] != -9 and (v[0] + v[1] != k[0]) and v != (0, 0, 0):
		print("Mom's genotype doesn't match (female X).", k, v)
	if k[1] != -9 and (2*v[2] != k[1]) and v != (0, 0, 0):
		print("Dad's genotype doesn't match (female X).", k, v)
	if k[2] != -9 and (v[0] + v[2] != k[2]) and v != (0, 0, 0):
		print("Child's genotype doesn't match (female X).", k, v)

# Pull arguments
vcf_file = sys.argv[1]
ped_file = sys.argv[2]
family_id = sys.argv[3]

start_time = time.time()

# Pull family structure from ped file
family = Family()
with open(ped_file, 'r') as f:
	for line in f:
		pieces = line.strip().split('\t')
		fam_id, child_id, father_id, mother_id = pieces[0:4]
		if fam_id == family_id:
			family.add_trio(child_id, mother_id, father_id)

# Pull data from vcf
with gzip.open(vcf_file, 'rt') as f:
	line = next(f)
	while line.startswith('##'):
		line = next(f)

	# Pull header
	pieces = line.strip().split('\t')
	for i, ind_id in enumerate(pieces[9:]):
		family.add_vcf_index(ind_id, i+9)
	line = next(f)
		
	# Load genotypes into numpy arrays
	n = len(family.members)
	gen_mapping = {b'./.': -9, b'0/0': -2, b'0/1': 0, b'1/0': 0, b'1/1': 2}
	converter = lambda gen:gen_mapping[gen[:3]]
	vcf_indices = [x.vcf_index for x in family.members if x.vcf_index is not None]
	data = np.loadtxt(f, dtype=np.int8, converters=dict(zip(vcf_indices, [converter]*n)),
		delimiter='\t', usecols=vcf_indices).T
	data = data[np.asarray(family.order()), :] #reorder so that mom, dad, child1, child2, ...
	print('Full dataset', data.shape)

	# Remove rows with missing entries
	data = data[:, (data!=-9).all(axis=0)]
	print('Remove missing entries', data.shape)

	# Remove non-mendelian rows
	is_mendelian = np.apply_along_axis(lambda x: set(x[2:]).issubset(mendelian_inheritance_map[(x[0], x[1])]), 0, data)
	data = data[:, is_mendelian]
	print('Remove non-mendelian entries', data.shape)

	# Remove only heterozygous site
	only_hetero = np.apply_along_axis(lambda x: set(x).issubset({0}), 0, data)
	data = data[:, ~only_hetero]
	print('Remove only heterozygous entries', data.shape)

	# Rough phase
	phased = np.apply_along_axis(lambda x: phase_map[tuple(x)], 0, data[:3, :])
	print(data)
	print(phased)

	# Assign (Y)
	m,n = data.shape
	p = phased.shape[0]
	all_pairs = np.zeros((n, (p-1)*(p-2)))
	index = 0
	for i in range(p):
		for j in range(i+1, p):
			all_pairs[:, index] = phased[i, :]+phased[j, :]
			index += 1
	print(['m1.m2', 'm1.f1', 'm1.f2', 'm2.f1', 'm2.f2', 'f1.f2'])
	print((data/2).dot((all_pairs/2))/n)
	


print('Done in %.2f sec' % (time.time()-start_time))


