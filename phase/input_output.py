import numpy as np
from scipy import sparse
import scipy.stats
import random
from collections import defaultdict
from os import listdir
import json

# From GRCh37.p13 https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh37.p13
chrom_lengths37 = {
	'1': 249250621,
	'2': 243199373,
	'3': 198022430,
	'4': 191154276,
	'5': 180915260,
	'6': 171115067,
	'7': 159138663,
	'8': 146364022,
	'9': 141213431,
	'10': 135534747,
	'11': 135006516,
	'12': 133851895,
	'13': 115169878,
	'14': 107349540,
	'15': 102531392,
	'16': 90354753,
	'17': 81195210,
	'18': 78077248,
	'19': 59128983,
	'20': 63025520,
	'21': 48129895,
	'22': 51304566,
	'X': 155270560,
	'Y': 59373566
}

# From GRCh38.p13 https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh38.p13
chrom_lengths38 = {
	'1': 248956422,
	'2': 242193529,
	'3': 198295559,
	'4': 190214555,
	'5': 181538259,
	'6': 170805979,
	'7': 159345973,
	'8': 145138636,
	'9': 138394717,
	'10': 133797422,
	'11': 135086622,
	'12': 133275309,
	'13': 114364328,
	'14': 107043718,
	'15': 101991189,
	'16': 90338345,
	'17': 83257441,
	'18': 80373285,
	'19': 58617616,
	'20': 64444167,
	'21': 46709983,
	'22': 50818468,
	'X': 156040895,
	'Y': 57227415
}

class Family():
	def __init__(self, famkey):
		self.id = famkey
		self.parents_to_children = defaultdict(list)
		self.mat_ancestors = []
		self.pat_ancestors = []
		self.descendents = []
		self.ordered_couples = []
		self.individuals = []
		self.individual_to_index = dict()

	def add_child(self, child_id, mother_id, father_id, retain_order=False):
		if child_id in self.mat_ancestors:
			self.mat_ancestors.remove(child_id)
		if child_id in self.pat_ancestors:
			self.pat_ancestors.remove(child_id)

		if mother_id not in self.individuals:
			self.mat_ancestors.append(mother_id)
		if father_id not in self.individuals:
			self.pat_ancestors.append(father_id)
		self.parents_to_children[(mother_id, father_id)].append(child_id)
		if not retain_order:
			random.shuffle(self.parents_to_children[(mother_id, father_id)])

		self._reset_individuals()

	def prune(self, sample_ids):
		for sample_id in sample_ids:
			is_leaf = True
			parents = None
			for (mom, dad), children in self.parents_to_children.items():
				if mom == sample_id or dad == sample_id:
					is_leaf = False
				if sample_id in children:
					parents = (mom, dad)

			if is_leaf:
				self.parents_to_children[parents].remove(sample_id)
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
				print(self.id)
				raise Exception('Circular pedigree.')
		self.individuals = self.mat_ancestors + self.pat_ancestors + self.descendents
		self.individual_to_index = dict([(x, i) for i, x in enumerate(self.individuals)])


	def set_individual_order(self, individuals):	
		assert sorted(individuals) == sorted(self.individuals)	
		self.mat_ancestors = individuals[:len(self.mat_ancestors)]
		self.pat_ancestors = individuals[len(self.mat_ancestors):len(self.mat_ancestors)+len(self.pat_ancestors)]
		self.descendents = []
		self.ordered_couples = []

		descendents = individuals[self.num_ancestors():]
		while len(descendents)>0:
			for (mom, dad), children in self.parents_to_children.items():
				if descendents[0] in children:
					self.parents_to_children[(mom, dad)] = descendents[:len(children)]
					descendents = descendents[len(children):]
					self.ordered_couples.append((mom, dad))
					self.descendents.extend(self.parents_to_children[(mom, dad)])
					break
		
		self.individuals = self.mat_ancestors + self.pat_ancestors + self.descendents
		self.individual_to_index = dict([(x, i) for i, x in enumerate(self.individuals)])

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

	def ind_filter(self, individuals):
		ind_filter = np.zeros((len(self),), dtype=bool)
		ind_filter[[self.individual_to_index[ind] for ind in individuals]] = True
		return ind_filter 



def pull_families(ped_file, retain_order=False):
	# pull families from ped file
	families = dict()
	with open(ped_file, 'r') as f:	
		for line in f:
			pieces = line.strip().split('\t')
			if len(pieces) < 4:
				print('ped parsing error', line)
			else:
				fam_id, child_id, f_id, m_id = pieces[0:4]
				child_id = child_id.replace('.', '_')
				f_id = f_id.replace('.', '_')
				m_id = m_id.replace('.', '_')

				if f_id != '0' and m_id != '0':
					if fam_id not in families:
						families[fam_id] = Family(fam_id)
					families[fam_id].add_child(child_id, m_id, f_id, retain_order)
	families = sorted([x for x in families.values()])
		
	print('families pulled %d' % len(families))
	return families

def pull_families_missing_parent(ped_file, data_dir, retain_order=False):
	with open('%s/genotypes/samples.json' % data_dir, 'r') as f:
		samples = set(json.load(f))

	# pull families from ped file that are missing a single parent
	families = dict()
	with open(ped_file, 'r') as f:	
		for line in f:
			pieces = line.strip().split('\t')
			if len(pieces) < 4:
				print('ped parsing error', line)
			else:
				fam_id, child_id, f_id, m_id = pieces[0:4]

				if child_id in samples and ((m_id in samples and f_id not in samples) or (m_id in samples and f_id not in samples)):
					if fam_id not in families:
						families[fam_id] = Family(fam_id)
					families[fam_id].add_child(child_id, m_id, f_id, retain_order)
	families = sorted([x for x in families.values() if len(x)==4])
		
	print('families pulled %d' % len(families))
	return families

def pull_families_from_file(fam_file):
	families = []
	with open(fam_file, 'r') as f:
		next(f) # skip header
		for line in f:
			pieces = line.strip().split('\t')
			families.append((pieces[0], pieces[1:]))
	return families

def pull_phase(phase_file, famkey, m, n):
	state_len = 2*m
	states = np.zeros((state_len, n), dtype=np.int8)
	with open(phase_file, 'r') as f:
		next(f) # skip header
		for line in f:
			pieces = line.strip().split('\t')
			if pieces[0] == famkey:
				# this is our family of interest
				start, end = int(pieces[-4]), int(pieces[-3])+1
				states[:, start:end] = np.tile([int(x) for x in pieces[1:(state_len+1)]], (end-start, 1)).T
	return states

def pull_sex(ped_file):
	sample_id_to_sex = dict()
	with open(ped_file, 'r') as f:	
		for line in f:
			pieces = line.strip().split('\t')
			if len(pieces) >= 5:
				fam_id, child_id, f_id, m_id, sex, aff = pieces[:6]
				sample_id_to_sex[child_id] = sex
	return sample_id_to_sex

def pull_phenotype(ped_file):
	sample_id_to_aff = dict()
	with open(ped_file, 'r') as f:	
		for line in f:
			pieces = line.strip().split('\t')
			if len(pieces) >= 6:
				fam_id, child_id, f_id, m_id, sex, aff = pieces[:6]
				sample_id_to_aff[child_id] = aff
	return sample_id_to_aff

def pull_gen_data_for_individuals(data_dir, assembly, chrom, individuals, start_pos=None, end_pos=None, use_pass=True):
	
	sample_file = '%s/genotypes/samples.json' % data_dir
	# pull samples
	with open(sample_file, 'r') as f:
		sample_ids = json.load(f)
	sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])


	# pull coordinates
	# use only SNPs, no indels
	# use only variants that PASS GATK
	# pull data only for individuals
	m = len(individuals)
	has_seq = np.array(np.where([x in sample_id_to_index for x in individuals])[0].tolist())
	ind_indices = [sample_id_to_index[x] for x in individuals if x in sample_id_to_index]

	if len(ind_indices)==0:
		return np.zeros((m, 0)), np.zeros((0, 2)), np.zeros((0,)) 


	gen_files = sorted([f for f in listdir('%s/genotypes' % data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f], key=lambda x: int(x.split('.')[2]))
	coord_files = sorted([f for f in listdir('%s/genotypes' % data_dir) if ('chr.%s.' % chrom) in f and 'gen.coordinates.npy' in f], key=lambda x: int(x.split('.')[2]))
	#af_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.af.npy' in f], key=lambda x: int(x.split('.')[2]))

	#print(len(gen_files), len(coord_files))
	assert len(gen_files) == len(coord_files)
	#assert len(gen_files) == len(af_files)

	# pull chrom length
	assert assembly == '37' or assembly == '38'
	chrom_length = chrom_lengths37[chrom] if assembly == '37' else chrom_lengths38[chrom] if assembly == '38' else None

	gens, snp_positions, collapseds = [], [], []
	total_pos = 0
	for gen_file, coord_file in zip(gen_files, coord_files):
		coords = np.load('%s/genotypes/%s' % (data_dir, coord_file))

		if coords.shape[0]>0:
			poss = coords[:, 1]
			is_snp = coords[:, 2]==1
			is_pass = coords[:, 3]==1

			if not use_pass:
				is_pass = np.ones(is_pass.shape, dtype=bool)

			# remove multiallelic sites
			multi_indices = np.where(coords[1:, 1]==coords[:-1, 1])[0]
			is_pass[multi_indices] = False
			is_pass[multi_indices+1] = False

			if start_pos is not None and end_pos is not None:
				in_interval = (coords[:, 1]>=start_pos) & (coords[:, 1]<=end_pos)
			else:
				in_interval = np.ones((is_snp.shape[0],), dtype=bool)

			if np.sum(is_snp & is_pass & in_interval)>0:
				gen = sparse.load_npz('%s/genotypes/%s' % (data_dir, gen_file))[ind_indices, :]
				total_pos += np.sum(is_snp & is_pass)
				family_has_variant = ((gen>0).sum(axis=0)>0).A.flatten()

				has_data = np.where(is_snp & is_pass & in_interval & family_has_variant)[0]

				if len(has_data)>0:
				
					# count the number of observed sites in between snps with data
					c = np.cumsum(is_snp & is_pass & in_interval & ~family_has_variant)
					collapsed = np.zeros((len(has_data),), dtype=int)
					collapsed_front = c[has_data[0]]
					collapsed[:-1] = c[has_data][1:]-c[has_data][:-1]
					collapsed[-1] = c[-1]-c[has_data[-1]]
					#print(c[-1]+len(has_data), np.sum(is_snp & is_pass), np.sum(collapsed)+len(has_data)+collapsed_front)

					if len(collapseds) == 0:
						collapseds.append([collapsed_front])
					else:
						collapseds[-1][-1] += collapsed_front

					gens.append(gen[:, has_data].A)
					snp_positions.append(poss[has_data])
					#afs.append(np.digitize(-np.log10(np.clip(af[has_data], 10**-(af_boundaries[0]+1), None)), af_boundaries))
					collapseds.append(collapsed)

	if len(gens)== 0:
		return np.zeros((m, 0)), np.zeros((0, 2)), np.zeros((0,))

	gens = np.hstack(gens)
	snp_positions = np.hstack(snp_positions)
	collapseds = np.hstack(collapseds)
	print(gens.shape, snp_positions.shape, collapseds.shape)

	assert np.all(snp_positions <= chrom_length)
	assert np.all(snp_positions[1:]>=snp_positions[:-1])
	assert np.all(collapseds >= 0)

	n = 2*len(snp_positions)+1
	family_genotypes = np.zeros((len(has_seq), n), dtype=np.int8)
	family_genotypes[:, np.arange(1, n-1, 2)] = gens
		
	observed = np.zeros((n,), dtype=int)
	observed[np.arange(1, n-1, 2)] = 1
	observed[np.arange(0, n, 2)] = collapseds
		
	family_snp_positions = np.zeros((n, 2), dtype=np.int)
	family_snp_positions[np.arange(1, n-1, 2), 0] = snp_positions
	family_snp_positions[np.arange(1, n-1, 2), 1] = snp_positions+1

	family_snp_positions[np.arange(0, n-2, 2), 1] = snp_positions
	family_snp_positions[np.arange(2, n, 2), 0] = snp_positions+1
	family_snp_positions[0, 0] = 1
	family_snp_positions[-1, 1] = chrom_length

	assert np.all(family_snp_positions[:, 1] >= family_snp_positions[:, 0])

	# remove unnecessary intervals
	haslength = np.where(family_snp_positions[:, 0]!=family_snp_positions[:, 1])[0]
	family_genotypes = family_genotypes[:, haslength]
	family_snp_positions = family_snp_positions[haslength, :]
	observed = observed[haslength]

	# aggregate identical genotypes
	rep_indices = np.where(np.any(family_genotypes[:, 1:]!=family_genotypes[:, :-1], axis=0))[0]
	n = rep_indices.shape[0]+1
	#print('n', n)

	new_family_genotypes = np.zeros((m, n), dtype=np.int8)
	mult_factor = np.zeros((n,), dtype=np.int)

	new_family_genotypes[np.ix_(has_seq, np.arange(n-1))] = family_genotypes[:, rep_indices]
	new_family_genotypes[has_seq, -1] = family_genotypes[:, -1]

	new_family_snp_positions = np.zeros((n, 2), dtype=np.int)
	new_family_snp_positions[0, 0] = family_snp_positions[0, 0]
	new_family_snp_positions[:-1, 1] = family_snp_positions[rep_indices, 1]
	new_family_snp_positions[1:, 0] = family_snp_positions[rep_indices+1, 0]
	new_family_snp_positions[-1, 1] = family_snp_positions[-1, 1]

	#print(new_family_snp_positions)

	c = np.cumsum(observed)
	if len(rep_indices)==0:
		mult_factor[0] = np.sum(observed)
	else:
		mult_factor[0] = c[rep_indices[0]]
		mult_factor[1:-1] = c[rep_indices[1:]] - c[rep_indices[:-1]]
		mult_factor[-1] = c[-1] - c[rep_indices[-1]]
	#print(c[-1], np.sum(mult_factor))
	#assert np.all(mult_factor>=0)

	print('genotypes pulled', new_family_genotypes.shape)
	#print(mult_factor)

	return new_family_genotypes, new_family_snp_positions, mult_factor

def write_to_file(phasef, chrom, family, final_states, family_snp_positions, cost):
	# write final states to file
	change_indices = [-1] + np.where(np.any(final_states[:, 1:]!=final_states[:, :-1], axis=0))[0].tolist() + [family_snp_positions.shape[0]-1]
	for j in range(1, len(change_indices)):
		s_start, s_end = change_indices[j-1]+1, change_indices[j]
		#assert np.all(final_states[:, s_start] == final_states[:, s_end])
		phasef.write('%s\t%d\t%d\t%s\n' % (
					'chr' + chrom, family_snp_positions[s_start, 0], family_snp_positions[s_end, 1],
					'\t'.join(map(str, final_states[:, s_start]))))
	print('Write to file complete')

class PhasedSegment():
	def __init__(self, chrom, start_pos, end_pos, deletions, phase, loss_region):
		self.chrom = chrom
		self.start_pos = start_pos
		self.end_pos = end_pos
		self.deletions = deletions
		self.mat_phase = [phase[i] for i in range(0, len(phase), 2)]
		self.pat_phase = [phase[i] for i in range(1, len(phase), 2)]
		self.loss_region = loss_region

	def is_mat_upd(self, index=None):
		if index is None:
			return [self.is_mat_upd(i) for i in range(len(self.mat_phase))]
		else:
			return self.mat_phase[index]==2 or self.mat_phase[index]==3

	def is_pat_upd(self, index=None):
		if index is None:
			return [self.is_pat_upd(i) for i in range(len(self.pat_phase))]
		else:
			return self.pat_phase[index]==0 or self.pat_phase[index]==1

	def is_hts(self):
		if self.chrom == 'X':
			return self.loss_region==1 or self.loss_region==3
		else:
			return self.loss_region==1

	def length(self):
		return self.end_pos-self.start_pos

def parse_phase_file(phase_file, chroms=None):
	with open(phase_file, 'r')  as f:
		next(f) # skip description
		family_members = next(f)[1:].strip().split(',')
		header = next(f).strip().split('\t')
		num_dels = len([x for x in header[3:] if x.endswith('_del')])
		is_standard_family_structure = num_dels == 4

		yield family_members, is_standard_family_structure

		runtimes = {}
		for line in f:
			if chroms is None or line.split('\t', maxsplit=1)[0][4 if line.startswith('#') else 3:] in chroms:
				pieces = line.strip().split('\t')
				chrom = pieces[0][3:]
				start_pos = int(pieces[1])
				end_pos = int(pieces[2])
				state_space = [int(x) for x in pieces[3:]]
				yield PhasedSegment(chrom, start_pos, end_pos, state_space[:num_dels], state_space[num_dels:-1], state_space[-1])


def pull_states_from_file(phasef, chrom, family_snp_positions):
	coords = []
	states = []
	next(phasef) # skip header
	for line in phasef:
		pieces = line.strip().split('\t')
		if pieces[0][3:] == chrom:
			states.append([int(x) for x in pieces[1:-2]])
			coords.append([int(x) for x in pieces[-2:]])

	coords = np.array(coords)
	states = np.array(states).T

	return states[:, np.searchsorted(coords[:, 0], family_snp_positions[:, 0], side='right')-1]


def convert_to_csr(A, family_snp_positions, mult_factor, chrom_length):
	nonzeros = np.sum(mult_factor*np.sum(A!=0, axis=0))
	data = np.zeros((nonzeros,), dtype=A.dtype)
	row_ind = np.zeros((nonzeros,), dtype=int)
	col_ind = np.zeros((nonzeros,), dtype=int)

	data_index = 0
	for i, j in zip(*np.nonzero(A)):
		pos_start, pos_end = family_snp_positions[j, :]
		pos_length = pos_end - pos_start
		
		data[data_index:(data_index+pos_length)] = A[i, j]
		row_ind[data_index:(data_index+pos_length)] = i
		col_ind[data_index:(data_index+pos_length)] = range(pos_start, pos_end)
		data_index += pos_length

	return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(A.shape[0], chrom_length), dtype=A.dtype)
