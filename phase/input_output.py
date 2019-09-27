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

	        	if child_id in sample_ids and f_id in sample_ids and m_id in sample_ids:
	        		if (fam_id, m_id, f_id) not in families:
	        			families[(fam_id, m_id, f_id)] = [m_id, f_id]
	        		families[(fam_id, m_id, f_id)].append(child_id)

	# randomly permute children
	families = dict([(k, x[:2]+random.sample(x[2:], len(x)-2)) for k, x in families.items()])
	print('families with sequence data', len(families))

	families_of_this_size = sorted([(fkey, inds) for fkey, inds in families.items() if len(inds) == m], key=lambda x: x[0])
	print('families of size %d: %d' % (m, len(families_of_this_size)))

	# limit to batch
	if batch_size is not None:
		families_of_this_size = families_of_this_size[batch_offset:(batch_size+batch_offset)]
		
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

		# use only SNPs, no indels
		# use only variants that PASS GATK
		coordinates = np.load(coord_file)
		self.snp_positions = coordinates[:, 1]
		self.is_snp = coordinates[:, 2]==1
		self.is_pass = coordinates[:, 3]==1

		self.snp_positions = self.snp_positions[self.is_snp & self.is_pass]
		print('chrom shape only SNPs', self.snp_positions.shape)

	
	def pull_data_for_individuals(self, individuals):
		#load from npz
		m = len(individuals)
		ind_indices = [self.sample_id_to_index[x] for x in individuals]
		
		data = sparse.hstack([sparse.load_npz('%s/%s' % (self.data_dir, gen_file))[ind_indices, :] for gen_file in self.gen_files]).A
		data = data[:, self.is_snp & self.is_pass]

		data[data<0] = -1

		print('% all homref', np.sum(np.all(data==0, axis=0))/data.shape[1])
		print('% all homref or missing', np.sum(np.all(data<=0, axis=0))/data.shape[1])

		n = 2*self.snp_positions.shape[0]+1
		family_genotypes = np.zeros((m, n), dtype=np.int8)
		family_genotypes[:, np.arange(1, n-1, 2)] = data
		observed = np.zeros((n,), dtype=bool)
		observed[np.arange(1, n-1, 2)] = True
		
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
		observed = observed[haslength]

		# aggregate identical genotypes
		rep_indices = np.where(np.any(family_genotypes[:, 1:]!=family_genotypes[:, :-1], axis=0))[0]
		n = rep_indices.shape[0]+1
		print('n', n)

		new_family_genotypes = np.zeros((m, n), dtype=np.int8)
		mult_factor = np.zeros((n,), dtype=np.int)

		new_family_genotypes[:, :-1] = family_genotypes[:, rep_indices]
		new_family_genotypes[:, -1] = family_genotypes[:, -1]

		new_family_snp_positions = np.zeros((n, 2), dtype=np.int)
		new_family_snp_positions[0, 0] = family_snp_positions[0, 0]
		new_family_snp_positions[:-1, 1] = family_snp_positions[rep_indices, 1]
		new_family_snp_positions[1:, 0] = family_snp_positions[rep_indices+1, 0]
		new_family_snp_positions[-1, 1] = family_snp_positions[-1, 1]

		c = np.cumsum(observed)
		mult_factor[0] = c[rep_indices[0]]
		mult_factor[1:-1] = c[rep_indices[1:]] - c[rep_indices[:-1]]
		mult_factor[-1] = c[-1] - c[rep_indices[-1]]

		#assert np.all(new_family_genotypes[:, mult_factor>10]==0)

		return new_family_genotypes, new_family_snp_positions, mult_factor

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
