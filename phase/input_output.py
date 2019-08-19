import numpy as np
from scipy import sparse
import scipy.stats
import random

# From GRCh37.p13 https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh37.p13
chrom_lengths = {
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

def pull_families(sample_file, ped_file, m, batch_size=None, batch_offset=None):
	# pull families with sequence data
	with open(sample_file, 'r') as f:
		sample_ids = [line.strip() for line in f]

	# pull families from ped file
	families = dict()
	with open(ped_file, 'r') as f:	
	    for line in f:
	        pieces = line.strip().split('\t')
	        if len(pieces) < 4:
	        	print('ped parsing error', line)
	        else:
	        	fam_id, child_id, f_id, m_id = pieces[0:4]

	        	if pieces[4] == '1' and child_id in sample_ids and f_id in sample_ids and m_id in sample_ids and 'LCL' not in child_id:
	        		if (fam_id, m_id, f_id) not in families:
	        			families[(fam_id, m_id, f_id)] = [m_id, f_id]
	        		families[(fam_id, m_id, f_id)].append(child_id)

	# randomly permute children
	families = dict([(k, x[:2]+random.sample(x[2:], len(x)-2)) for k, x in families.items()])
	print('families with sequence data', len(families))

	families_of_this_size = [(fkey, inds) for fkey, inds in families.items() if len(inds) == m]
	print('families of size %d: %d' % (m, len(families_of_this_size)))

	# limit to batch
	if batch_size is not None:
		family_keys = set(sorted([x[0] for x in families_of_this_size])[batch_offset:(batch_size+batch_offset)])
		families_of_this_size = [(k, v) for k, v in families_of_this_size if k in family_keys]
	
	print('families pulled %d: %d' % (m, len(families_of_this_size)))
	return families_of_this_size

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

class WGSData:
	def __init__(self, data_dir, gen_files, coord_file, sample_file, ped_file, chrom):

		self.chrom_length = chrom_lengths[chrom]
		self.data_dir = data_dir
		self.gen_files = gen_files

		with open(sample_file, 'r') as f:
			sample_ids = [line.strip() for line in f]
		self.sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])

		# use only "cleaned" variants - must be SNPs
		coordinates = np.load(coord_file)
		self.snp_positions = coordinates[:, 1]
		self.snp_indices = coordinates[:, 2]==1

		self.snp_positions = self.snp_positions[self.snp_indices]
		print('chrom shape only SNPs', self.snp_positions.shape)

		# Test Hardy-Weinburg Equilibrium
		#self.__test_for_hardy_weinburg__(ped_file, chrom)
		self.pass_hw = np.ones((self.snp_positions.shape[0],), dtype=bool)


	def __test_for_hardy_weinburg__(self, ped_file, chrom):
		# test each position for hardy-weinburg equilibrium in the presence of deletions

		# pull parent_indices
		parent_indices = set()
		with open(ped_file, 'r') as f:
			for line in f:
				pieces = line.strip().split('\t')
				if len(pieces) >= 6:
					fam_id, child_id, f_id, m_id, sex, disease_status = pieces[0:6]
					if chrom != 'X' and f_id in self.sample_id_to_index:
						parent_indices.add(self.sample_id_to_index[f_id])
					if m_id in self.sample_id_to_index:
						parent_indices.add(self.sample_id_to_index[m_id])
		parent_indices = list(parent_indices)

		# pull genotype counts for parents
		parent_gen_counts = []
		for gen_file in self.gen_files:
			gen_data = sparse.load_npz('%s/%s' % (self.data_dir, gen_file))

			gen_counts = np.zeros((4, gen_data.shape[1]), dtype=int)
			gen_counts[1, :] = (gen_data[parent_indices, :]==1).sum(axis=0)
			gen_counts[2, :] = (gen_data[parent_indices, :]==2).sum(axis=0)
			gen_counts[3, :] = (gen_data[parent_indices, :]<0).sum(axis=0)
			gen_counts[0, :] = len(parent_indices) - np.sum(gen_counts, axis=0)
			gen_counts = gen_counts/len(parent_indices) # normalize

			parent_gen_counts.append(gen_counts)
		parent_gen_counts = np.hstack(parent_gen_counts)
		parent_gen_counts = parent_gen_counts[:, self.snp_indices]

		# r = deletion frequency, p = ref allele frequency, q = alt allele frequency
		# estimate p, q, r based on Hardy-Weinburg Equilibrium
		r = np.sqrt(parent_gen_counts[3, :])
		p = -r + np.sqrt(np.power(r, 2) + parent_gen_counts[0, :])
		q = 1-r-p

		# expected values based on HW
		exp = np.zeros(parent_gen_counts.shape, dtype=float)
		exp[0, :] = np.power(p, 2) + 2*p*r
		exp[1, :] = 2*p*q
		exp[2, :] = np.power(q, 2) + 2*q*r
		exp[3, :] = np.power(r, 2)

		# use chi-square test to test for deviations from equilibrium
		chisq, pvalue = scipy.stats.chisquare(len(parent_indices)*parent_gen_counts[1:3, :], len(parent_indices)*exp[1:3, :])

		self.pass_hw = ~np.isnan(pvalue) & (pvalue>(0.05/parent_gen_counts.shape[1])) & (q > 0.01)
		print('% positions passing HW for deletions', np.sum(self.pass_hw)/self.pass_hw.shape[0])
		print('% missing passing HW for deletions', np.sum(parent_gen_counts[3, self.pass_hw])/np.sum(parent_gen_counts[3, :]))

	def pull_data_for_individuals(self, individuals):
		#load from npz
		m = len(individuals)
		ind_indices = [self.sample_id_to_index[x] for x in individuals]
		
		data = sparse.hstack([sparse.load_npz('%s/%s' % (self.data_dir, gen_file))[ind_indices, :] for gen_file in self.gen_files]).A
		data = data[:, self.snp_indices]

		data[data<0] = -1

		print('% all homref', np.sum(np.all(data==0, axis=0))/data.shape[1])
		print('% all homref or missing', np.sum(np.all(data<=0, axis=0))/data.shape[1])

		# -1 indicates a missing value due to hard to sequence region, etc
		# -2 indicates no information (not in VCF)
		# -3 indicates a potential double deletion
		data[np.tile(self.pass_hw, (m, 1)) & (data==-1)] = -3

		n = 2*self.snp_positions.shape[0]+1
		#family_genotypes = -2*np.ones((m, n), dtype=np.int8)
		family_genotypes = np.zeros((m, n), dtype=np.int8)
		family_genotypes[:, np.arange(1, n-1, 2)] = data
		

		family_snp_positions = np.zeros((n, 2), dtype=np.int)
		family_snp_positions[0, 0] = 0
		family_snp_positions[np.arange(0, n-2, 2), 1] = self.snp_positions-1
		family_snp_positions[np.arange(1, n-1, 2), 0] = self.snp_positions-1
		family_snp_positions[np.arange(1, n-1, 2), 1] = self.snp_positions
		family_snp_positions[np.arange(2, n, 2), 0] = self.snp_positions
		family_snp_positions[-1, 1] = self.chrom_length

		# remove unnecessary intervals
		haslength = np.where(family_snp_positions[:, 0]!=family_snp_positions[:, 1])[0]
		family_genotypes = family_genotypes[:, haslength]
		family_snp_positions = family_snp_positions[haslength, :]

		# aggregate identical genotypes
		rep_indices = np.where(np.any(family_genotypes[:, 1:]!=family_genotypes[:, :-1], axis=0))[0]
		n = rep_indices.shape[0]+1
		print('n', n)

		new_family_genotypes = np.zeros((m, n), dtype=np.int8)
		new_family_genotypes[:, :-1] = family_genotypes[:, rep_indices]
		new_family_genotypes[:, -1] = family_genotypes[:, -1]

		new_family_snp_positions = np.zeros((n, 2), dtype=np.int)
		new_family_snp_positions[0, 0] = family_snp_positions[0, 0]
		new_family_snp_positions[:-1, 1] = family_snp_positions[rep_indices, 1]
		new_family_snp_positions[1:, 0] = family_snp_positions[rep_indices+1, 0]
		new_family_snp_positions[-1, 1] = family_snp_positions[-1, 1]

		family_genotypes, family_snp_positions = new_family_genotypes, new_family_snp_positions

		mult_factor = family_snp_positions[:, 1] - family_snp_positions[:, 0]

		return family_genotypes, family_snp_positions, mult_factor

def write_to_file(famf, statef, fkey, individuals, final_states, family_snp_positions):
	# write family to file
	famf.write('%s\t%s\n' % ('.'.join(fkey), '\t'.join(individuals)))
	famf.flush()

	# write final states to file
	change_indices = [-1] + np.where(np.any(final_states[:, 1:]!=final_states[:, :-1], axis=0))[0].tolist()
	for j in range(1, len(change_indices)):
		s_start, s_end = change_indices[j-1]+1, change_indices[j]
		#assert np.all(final_states[:, s_start] == final_states[:, s_end])
		statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
					'.'.join(fkey), 
					'\t'.join(map(str, final_states[:, s_start])), 
					family_snp_positions[s_start, 0]+1, family_snp_positions[s_end, 1],
					s_start, s_end, 
					family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0], 
					s_end-s_start+1))

	# last entry
	s_start, s_end = change_indices[-1]+1, family_snp_positions.shape[0]-1
	#assert np.all(final_states[:, s_start] == final_states[:, s_end])
	statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
				'.'.join(fkey), 
				'\t'.join(map(str, final_states[:, s_start])), 
				family_snp_positions[s_start, 0]+1, family_snp_positions[s_end, 1],
				s_start, s_end, 
				family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0]+1, 
				s_end-s_start+1))
	statef.flush()	

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
