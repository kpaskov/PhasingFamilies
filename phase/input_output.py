import numpy as np
from scipy import sparse
import scipy.stats
import random
from collections import defaultdict
from os import listdir

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
				fam_id, child_id, f_id, m_id, sex = pieces[:5]
				sample_id_to_sex[child_id] = sex
	return sample_id_to_sex

def pull_gen_data_for_individuals(data_dir, af_boundaries, assembly, chrom, individuals):
	gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f], key=lambda x: 0 if len(x.split('.'))==4 else int(x.split('.')[2]))
	coord_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.coordinates.npy' in f], key=lambda x: 0 if len(x.split('.'))==5 else int(x.split('.')[2]))
	af_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.af.npy' in f], key=lambda x: 0 if len(x.split('.'))==5 else int(x.split('.')[2]))
	sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)

	# pull samples
	with open(sample_file, 'r') as f:
		sample_ids = [line.strip() for line in f]
	sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])

	# pull chrom length
	assert assembly == '37' or assembly == '38'
	chrom_length = chrom_lengths37[chrom] if assembly == '37' else chrom_lengths38[chrom] if assembly == '38' else None

	# pull coordinates
	# use only SNPs, no indels
	# use only variants that PASS GATK
	coordinates = [np.load('%s/%s' % (data_dir, coord_file)) for coord_file in coord_files]
	has_data = np.where([x.shape[0]>0 for x in coordinates])[0]
	coordinates = np.vstack([coordinates[i] for i in has_data])
	snp_positions = coordinates[:, 1]
	is_snp = coordinates[:, 2]==1
	is_pass = coordinates[:, 3]==1

	snp_positions = snp_positions[is_snp & is_pass]
	assert np.all(snp_positions <= chrom_length)
	#print('chrom shape only SNPs', snp_positions.shape)

	# pull af
	af = np.hstack([np.load('%s/%s' % (data_dir, af_files[i])) for i in has_data]).flatten()
	af = np.digitize(-np.log10(af[is_snp & is_pass]), af_boundaries)

	# pull genotypes
	m = len(individuals)
	has_seq = np.array(np.where([x in sample_id_to_index for x in individuals])[0].tolist() + [m])
	ind_indices = [sample_id_to_index[x] for x in individuals if x in sample_id_to_index]
		
	data = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_files[i]))[ind_indices, :] for i in has_data]).A
	data = data[:, is_snp & is_pass]

	data[data<0] = -1

	# append af to end of family genotypes
	print(data.shape, af.shape)
	data = np.vstack((data, af))

	# if a position has only 0s and -1s in the family, assume it's homref for everyone
	# we don't use af for sites where the whole family is homref, so set af to the 0 bin for efficiency
	data[:, np.all(data[:-1, :]<=0, axis=0)] = 0

	#print('all homref', np.sum(np.all(data==0, axis=0))/data.shape[1])
	#print('all homref or missing', np.sum(np.all(data<=0, axis=0))/data.shape[1])

	# remove multiallelic sites
	is_multiallelic = np.zeros((snp_positions.shape[0],), dtype=bool)
	indices = np.where(snp_positions[:-1] == snp_positions[1:])[0]
	is_multiallelic[indices] = True
	is_multiallelic[indices+1] = True

	n = 2*np.sum(~is_multiallelic)+1
	family_genotypes = np.zeros((len(has_seq), n), dtype=np.int8)
	family_genotypes[:, np.arange(1, n-1, 2)] = data[:, ~is_multiallelic]
		
	observed = np.zeros((n,), dtype=bool)
	observed[np.arange(1, n-1, 2)] = True
		
	family_snp_positions = np.zeros((n, 2), dtype=np.int)
	family_snp_positions[np.arange(1, n-1, 2), 0] = snp_positions[~is_multiallelic]
	family_snp_positions[np.arange(1, n-1, 2), 1] = snp_positions[~is_multiallelic]+1

	family_snp_positions[np.arange(0, n-2, 2), 1] = snp_positions[~is_multiallelic]
	family_snp_positions[np.arange(2, n, 2), 0] = snp_positions[~is_multiallelic]+1
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

	new_family_genotypes = np.zeros((m+1, n), dtype=np.int8)
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
	mult_factor[0] = c[rep_indices[0]]
	mult_factor[1:-1] = c[rep_indices[1:]] - c[rep_indices[:-1]]
	mult_factor[-1] = c[-1] - c[rep_indices[-1]]

	print('genotypes pulled', new_family_genotypes.shape, new_family_genotypes)
	#print(mult_factor)

	return new_family_genotypes, new_family_snp_positions, mult_factor

def write_to_file(phasef, chrom, family, final_states, family_snp_positions):
	# write final states to file
	change_indices = [-1] + np.where(np.any(final_states[:, 1:]!=final_states[:, :-1], axis=0))[0].tolist() + [family_snp_positions.shape[0]-1]
	for j in range(1, len(change_indices)):
		s_start, s_end = change_indices[j-1]+1, change_indices[j]
		#assert np.all(final_states[:, s_start] == final_states[:, s_end])
		phasef.write('%s\t%s\t%d\t%d\n' % (
					'chr' + chrom, 
					'\t'.join(map(str, final_states[:, s_start])), 
					family_snp_positions[s_start, 0], family_snp_positions[s_end, 1]))

	print('Write to file complete')

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
