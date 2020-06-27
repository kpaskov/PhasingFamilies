import sys
from collections import defaultdict
import random
from os import listdir
from scipy import sparse
import numpy as np

data_dir = sys.argv[1]
ped_file = sys.argv[2]
out_dir = sys.argv[3]

chroms = [str(x) for x in range(1, 23)] + ['X', 'Y']

class Family():
	def __init__(self, famkey):
		self.id = famkey
		self.parents_to_children = defaultdict(list)
		self.mat_ancestors = []
		self.pat_ancestors = []
		self.descendents = []
		self.ordered_couples = []
		self.individuals = []

	def add_child(self, child_id, mother_id, father_id):
		if child_id in self.mat_ancestors:
			self.mat_ancestors.remove(child_id)
		if child_id in self.pat_ancestors:
			self.pat_ancestors.remove(child_id)

		if mother_id not in self.individuals:
			self.mat_ancestors.append(mother_id)
		if father_id not in self.individuals:
			self.pat_ancestors.append(father_id)
		self.parents_to_children[(mother_id, father_id)].append(child_id)
		random.shuffle(self.parents_to_children[(mother_id, father_id)])

		self._reset_individuals()

	def get_parents(self, child_id):
		for ((mom, dad), children) in self.parents_to_children:
			if child_id in children:
				return (mom, dad)
		return None

	def _reset_individuals(self):
		self.descendents = []
		self.ordered_couples = []
		parents = set(self.parents_to_children.keys())
		while len(parents) > 0:
			already_added = set()
			for mom, dad in parents:
				if (mom in self.mat_ancestors or mom in self.descendents) and (dad in self.pat_ancestors or dad in self.descendents):
					self.ordered_couples.append((mom, dad))
					self.descendents.extend(self.parents_to_children[(mom, dad)])
					already_added.add((mom, dad))
			parents = parents - already_added
			if len(already_added) == 0:
				raise Exception('Circular pedigree.')
		self.individuals = self.mat_ancestors + self.pat_ancestors + self.descendents

	def __lt__(self, other):
		return self.id < other.id

	def __eq__(self, other):
		return self.id == other.id

	def __hash__(self):
		return hash(self.id)

	def __len__(self):
		return len(self.individuals)

	def __str__(self):
		return self.id

	def num_ancestors(self):
		return len(self.mat_ancestors) + len(self.pat_ancestors)

	def num_descendents(self):
		return len(self.descendents)


def pull_families(ped_file):
	# pull families from ped file
	families = dict()
	with open(ped_file, 'r') as f:	
		for line in f:
			pieces = line.strip().split('\t')
			if len(pieces) < 4:
				print('ped parsing error', line)
			else:
				fam_id, child_id, f_id, m_id = pieces[0:4]

				if f_id != '0' and m_id != '0':
					if fam_id not in families:
						families[fam_id] = Family(fam_id)
					families[fam_id].add_child(child_id, m_id, f_id)
	families = sorted([x for x in families.values()])
		
	print('families pulled %d' % len(families))
	return families

families = pull_families(ped_file)
ancestors = sorted(sum([x.mat_ancestors + x.pat_ancestors for x in families], []))
print('ancestors', len(ancestors))

for chrom in chroms:
	# pull samples
	sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
	with open(sample_file, 'r') as f:
		sample_ids = [line.strip() for line in f]
	sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])
	ind_indices = [sample_id_to_index[x] for x in ancestors if x in sample_id_to_index]  

	# calculate AF among ancestors
	gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f], key=lambda x: int(x.split('.')[2]))

	for gen_file in gen_files:
		print(gen_file)
		batch_num = int(gen_file.split('.')[2])

		# pull genotypes		
		data = sparse.load_npz('%s/%s' % (data_dir, gen_file))[ind_indices, :]
		num_het = (data==1).sum(axis=0).A.flatten()
		num_homalt = (data==2).sum(axis=0).A.flatten()
		num_missing = (data<0).sum(axis=0).A.flatten()
		num_homref = len(ind_indices) - num_het - num_homalt - num_missing
		assert np.all(num_homref >= 0)

		af = np.zeros((data.shape[1],))
		all_missing = (num_homref + num_het + num_homalt) == 0
		af[~all_missing] = (num_het[~all_missing] + 2*num_homalt[~all_missing])/(2*num_homref[~all_missing] + 2*num_het[~all_missing] + 2*num_homalt[~all_missing])
		np.save('%s/chr.%s.%d.gen.af' % (out_dir, chrom, batch_num), af)




