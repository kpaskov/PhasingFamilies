from collections import defaultdict
import numpy as np
import scipy.stats
import sys

phase_dir = 'phased_ssc'
ped_file = 'data/ssc.ped'
chrom = sys.argv[1]
family_sizes = [4]
outfile = '%s/chr.%s.IST.pvalues' % (phase_dir, chrom)

# pull phenotype
sample_to_affected = dict()
with open(ped_file, 'r') as f:
	for line in f:
		pieces = line.strip().split('\t')
		sample_to_affected[pieces[1]] = pieces[5]

def pull_samples(phase_dir, chrom):
	# pull individuals
	sample_ids = set()
	family_to_inds = defaultdict(list)
	for family_size in family_sizes:
		with open('%s/chr.%s.familysize.%d.families.txt' % (phase_dir, chrom, family_size), 'r') as f:
			next(f) # skip header
			for line in f:
				pieces = line.strip().split('\t')
				sample_ids.update(pieces[3:])
				family_to_inds[pieces[0]] = pieces[1:]
                
	sample_ids = sorted(sample_ids)
	return sample_ids, family_to_inds

def pull_phase(phase_dir, chrom, sample_ids, family_to_inds):
	sample_id_to_index = dict([(x, i) for i, x in enumerate(sample_ids)])
    
	# pull positions
	positions = set()
	for family_size in family_sizes:
		with open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, family_size), 'r')  as f:
			next(f) # skip header

			for line in f:
				pieces = line.strip().split('\t')
				family_key = pieces[0]
				start_pos, end_pos = [int(x) for x in pieces[(6+2*family_size):(8+2*family_size)]]
				assert end_pos >= start_pos
	                
				positions.add(start_pos)
				positions.add(end_pos)
                
	positions = sorted(positions)
	position_to_index = dict([(x, i) for i, x in enumerate(positions)])
    
	# pull phase data
	# sample, position
	mat_phase_data = -np.ones((len(sample_id_to_index), len(position_to_index)), dtype=int)
	pat_phase_data = -np.ones((len(sample_id_to_index), len(position_to_index)), dtype=int)
	for family_size in family_sizes:
		with open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, family_size), 'r')  as f:
			next(f) # skip header

			for line in f:
				pieces = line.strip().split('\t')
				family_key = pieces[0]
				start_pos, end_pos = [int(x) for x in pieces[(6+2*family_size):(8+2*family_size)]]
				state = np.array([int(x) for x in pieces[1:(6+2*family_size)]])
				inds = family_to_inds[family_key]
				child_indices = [sample_id_to_index[x] for x in inds[2:]]

				if state[-1] == 0:
					for child_index, state_index in zip(child_indices, np.arange(8, 4+2*len(inds), 2)):
						mat_phase_data[child_index, (position_to_index[start_pos]):(position_to_index[end_pos]+1)] = state[state_index]
					for child_index, state_index in zip(child_indices, np.arange(9, 4+2*len(inds), 2)):  
						pat_phase_data[child_index, (position_to_index[start_pos]):(position_to_index[end_pos]+1)] = state[state_index]
	return np.array(positions), mat_phase_data, pat_phase_data

def generate_test_statistic(positions, mat_phase_data, pat_phase_data):
	# table of our test statistic
	# family_size, is_multiplex, max matching, position
	mat_st = np.zeros((max(family_sizes)-1, 2, max(family_sizes)-1, len(positions)))
	pat_st = np.zeros((max(family_sizes)-1, 2, max(family_sizes)-1, len(positions)))

	mat_match = mat_phase_data == np.tile([sample_to_affected[x]=='2' for x in sample_ids], 
	                                            (len(positions), 1)).T
	pat_match = pat_phase_data == (2+np.tile([sample_to_affected[x]=='2' for x in sample_ids], 
	                                            (len(positions), 1)).T)
	sample_id_to_index = dict([(x, i) for i, x in enumerate(sample_ids)])

	for family_key, inds in family_to_inds.items():
		child_indices = [sample_id_to_index[x] for x in inds[2:]]
		is_multiplex = sum([sample_to_affected[x]=='2' for x in inds[2:]]) > 1
	    
		no_missing = np.all(mat_phase_data[child_indices, :]>=0, axis=0)
		family_mat_match = np.sum(mat_match[child_indices, :], axis=0)
		max_family_mat_match = np.maximum(family_mat_match, len(inds[2:]) - family_mat_match)
		mat_st[len(inds[2:])*np.ones((np.sum(no_missing),), dtype=int), 
			   is_multiplex*np.ones((np.sum(no_missing),), dtype=int),
			   max_family_mat_match[no_missing], 
			   np.where(no_missing)[0]] += 1

		no_missing = np.all(pat_phase_data[child_indices, :]>=0, axis=0)
		family_pat_match = np.sum(pat_match[child_indices, :], axis=0)
		max_family_pat_match = np.maximum(family_pat_match, len(child_indices) - family_pat_match)
		pat_st[len(inds[2:])*np.ones((np.sum(no_missing),), dtype=int), 
			   is_multiplex*np.ones((np.sum(no_missing),), dtype=int),
			   max_family_pat_match[no_missing], 
			   np.where(no_missing)[0]] += 1

	return mat_st, pat_st

ptable = np.array([[0, 0, 0, 0, 0, 0],
				   [0, 1, 0, 0, 0, 0],
				   [0, 1/2, 1/2, 0, 0, 0],
				   [0, 0, 3/4, 1/4, 0, 0],
				   [0, 0, 3/8, 4/8, 1/8, 0],
				   [0, 0, 0, 10/16, 5/16, 1/16]])
def calculate_pvalue(obs):
	num_fams = np.sum(obs, axis=1)

	exp = np.zeros(obs.shape)
	for i in range(max(family_sizes)-1):
		exp[i, :] = num_fams[i]*ptable[i, :max(family_sizes)-1]
	indices = np.where(exp>0)
	return scipy.stats.chisquare(obs[indices], exp[indices], ddof=np.sum(num_fams>0)-1).pvalue	

def calculate_pvalues(mat_st, pat_st):
	# pos, simplex/mutiplex/all, mat/pat/combined
	pvalues = np.ones((mat_st.shape[3], 3, 3))
	for index in range(mat_st.shape[3]):
		# mat
		pvalues[index, 0, 0] = calculate_pvalue(mat_st[:, 0, :, index])
		pvalues[index, 1, 0] = calculate_pvalue(mat_st[:, 1, :, index])
		pvalues[index, 2, 0] = calculate_pvalue(np.sum(mat_st[:, :, :, index], axis=1))

		# pat
		pvalues[index, 0, 1] = calculate_pvalue(pat_st[:, 0, :, index])
		pvalues[index, 1, 1] = calculate_pvalue(pat_st[:, 1, :, index])
		pvalues[index, 2, 1] = calculate_pvalue(np.sum(pat_st[:, :, :, index], axis=1))
	    
		# both
		pvalues[index, 0, 2] = calculate_pvalue(mat_st[:, 0, :, index]+pat_st[:, 0, :, index])
		pvalues[index, 1, 2] = calculate_pvalue(mat_st[:, 1, :, index]+pat_st[:, 1, :, index])
		pvalues[index, 2, 2] = calculate_pvalue(np.sum(mat_st[:, :, :, index], axis=1)+np.sum(pat_st[:, :, :, index], axis=1))
	return pvalues

sample_ids, family_to_inds = pull_samples(phase_dir, chrom)
print('samples', len(sample_ids))
positions, mat_phase_data, pat_phase_data = pull_phase(phase_dir, chrom, sample_ids, family_to_inds)
print('positions', len(positions))
mat_st, pat_st = generate_test_statistic(positions, mat_phase_data, pat_phase_data)
print('IST statistic computed')
pvalues = calculate_pvalues(mat_st, pat_st)
print('pvalues computed')
np.save(outfile, pvalues)
np.save(outfile + '.positions', positions)
print('results saved to %s' % outfile)


