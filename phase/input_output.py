import numpy as np
from scipy import sparse
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
	        fam_id, child_id, f_id, m_id = pieces[0:4]

	        if child_id in sample_ids and f_id in sample_ids and m_id in sample_ids:
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
	def __init__(self, data_dir, gen_files, coord_file, sample_file, chrom):

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

	def pull_data_for_individuals(self, individuals):
		#load from npz
		m = len(individuals)
		ind_indices = [self.sample_id_to_index[x] for x in individuals]
		
		data = sparse.hstack([sparse.load_npz('%s/%s' % (self.data_dir, gen_file))[ind_indices, :] for gen_file in self.gen_files]).A
		data = data[:, self.snp_indices]

		n = 2*self.snp_positions.shape[0]+1
		family_genotypes = np.zeros((m, n), dtype=np.int8)
		family_genotypes[:, np.arange(1, n-1, 2)] = data
		family_genotypes[:, -2] = family_genotypes[:, -1]

		# if any family member is missing, set whole family to 0 - this has the effect of ignoring missing positions
		#family_genotypes[:, np.any(family_genotypes<0, axis=0)] = 0
		
		# if we see two missing entries in a row, mark the middle interval as possibly missing/possibly homref (-1)
		family_genotypes[family_genotypes<0] = -1
		for i in range(m):
			double_missing = np.where((data[i, 1:]==-1) & (data[i, :-1]==-1))[0]
			family_genotypes[i, (2*double_missing)+2] = -1

		family_snp_positions = np.zeros((n, 2), dtype=np.int)
		family_snp_positions[0, 0] = 0
		family_snp_positions[np.arange(0, n-2, 2), 1] = self.snp_positions-1
		family_snp_positions[np.arange(1, n-1, 2), 0] = self.snp_positions-1
		family_snp_positions[np.arange(1, n-1, 2), 1] = self.snp_positions
		family_snp_positions[np.arange(2, n, 2), 0] = self.snp_positions
		family_snp_positions[-1, 1] = self.chrom_length

		# remove unnecessary ref positions
		haslength = np.where(family_snp_positions[:, 0]!=family_snp_positions[:, 1])[0]
		family_genotypes = family_genotypes[:, haslength]
		family_snp_positions = family_snp_positions[haslength, :]

		# aggregate identical genotypes
		rep_indices = np.where(np.any(family_genotypes[:, 1:]!=family_genotypes[:, :-1], axis=0))[0]
		n = rep_indices.shape[0]+1

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