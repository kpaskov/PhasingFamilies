from collections import defaultdict
import numpy as np
import scipy.stats
import sys

chrom = sys.argv[1]

phase_dir = 'phased_spark'
ped_file = 'data/spark.ped'
family_sizes = [4, 5, 6, 7]

#phase_dir = 'phased_ssc'
#ped_file = 'data/ssc.ped'
#family_sizes = [4]

#phase_dir = 'phased_ihart'
#ped_file = 'data/v34.vcf.ped'
#family_sizes = [4]


interval = 1000000
outfile = '%s/chr.%s.IST.pvalues.intervals.%d' % (phase_dir, chrom, interval)

def pull_phenotype(ped_file):
	sample_to_sex = dict()
	sample_to_affected = dict()
	with open(ped_file, 'r') as f:
		for line in f:
			pieces = line.strip().split('\t')
			sample_to_sex[pieces[1]] = pieces[4]
			sample_to_affected[pieces[1]] = pieces[5]
	return sample_to_affected, sample_to_sex

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
	assert len(sample_ids) == sum([len(x[2:]) for x in family_to_inds.values()])
	return sample_ids, family_to_inds

def pull_phase(phase_dir, chrom, sample_ids, family_to_inds, interval):
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


	positions.update(np.arange(0, max(positions), interval))

	positions = np.array(sorted(positions))
	position_to_index = dict([(x, i) for i, x in enumerate(positions)])

	regions = np.hstack((positions[:-1, np.newaxis], positions[1:, np.newaxis]))
	print(regions.shape)

	# pull phase data
	# sample, position
	mat_phase_data = -np.ones((len(sample_id_to_index), regions.shape[0]), dtype=int)
	pat_phase_data = -np.ones((len(sample_id_to_index), regions.shape[0]), dtype=int)
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

				for child_index, state_index in zip(child_indices, np.arange(8, 4+2*len(inds), 2)):
					mat_phase_data[child_index, (position_to_index[start_pos]):(position_to_index[end_pos])] = state[state_index]
				for child_index, state_index in zip(child_indices, np.arange(9, 4+2*len(inds), 2)):  
					pat_phase_data[child_index, (position_to_index[start_pos]):(position_to_index[end_pos])] = state[state_index]
	return regions, mat_phase_data, pat_phase_data

def reduce_to_intervals(intervals, mat_phase_data, pat_phase_data, regions):
	mat_phase_data_interval = -np.ones((mat_phase_data.shape[0], intervals.shape[0]), dtype=int)
	pat_phase_data_interval = -np.ones((mat_phase_data.shape[0], intervals.shape[0]), dtype=int)

	for i, (interval_start, interval_end) in enumerate(intervals):
		overlap = np.minimum(regions[:, 1], interval_end) - np.maximum(regions[:, 0], interval_start)
		m1 = np.zeros((mat_phase_data.shape[0],), dtype=int)
		m2 = np.zeros((mat_phase_data.shape[0],), dtype=int)
		p1 = np.zeros((mat_phase_data.shape[0],), dtype=int)
		p2 = np.zeros((mat_phase_data.shape[0],), dtype=int)
		for region_index, overlap in zip(np.where(overlap>0)[0], overlap[overlap>0]):
			m1[mat_phase_data[:, region_index]==0] += overlap
			m2[mat_phase_data[:, region_index]==1] += overlap
			p1[pat_phase_data[:, region_index]==2] += overlap
			p2[pat_phase_data[:, region_index]==3] += overlap

		mat_phase_data_interval[m1>(interval_end-interval_start)*0.9, i] = 0
		mat_phase_data_interval[m2>(interval_end-interval_start)*0.9, i] = 1
		pat_phase_data_interval[p1>(interval_end-interval_start)*0.9, i] = 2
		pat_phase_data_interval[p2>(interval_end-interval_start)*0.9, i] = 3

		#mat_phase_data_interval[np.all((mat_phase_data[:, ok_positions] == 0) | (mat_phase_data[:, ok_positions] == -1), axis=1), i] = 0
		#mat_phase_data_interval[np.all((mat_phase_data[:, ok_positions] == 1) | (mat_phase_data[:, ok_positions] == -1), axis=1), i] = 1
		#mat_phase_data_interval[np.all(mat_phase_data[:, ok_positions] == -1, axis=1), i] = -1

		#pat_phase_data_interval[np.all((pat_phase_data[:, ok_positions] == 2) | (mat_phase_data[:, ok_positions] == -1), axis=1), i] = 2
		#pat_phase_data_interval[np.all((pat_phase_data[:, ok_positions] == 3) | (mat_phase_data[:, ok_positions] == -1), axis=1), i] = 3
		#pat_phase_data_interval[np.all(pat_phase_data[:, ok_positions] == -1, axis=1), i] = -1

	return mat_phase_data_interval, pat_phase_data_interval

def generate_test_statistic(mat_phase_data, pat_phase_data, sample_ids, sample_to_affected, sample_to_sex):
	# table of our test statistic
	# family_size, simplexF/simplexM/multiplex, max matching, position
	mat_st = np.zeros((max(family_sizes)-1, 3, max(family_sizes)-1, mat_phase_data.shape[1]), dtype=int)
	pat_st = np.zeros((max(family_sizes)-1, 3, max(family_sizes)-1, pat_phase_data.shape[1]), dtype=int)

	mat_match = mat_phase_data == np.tile([sample_to_affected[x]=='2' for x in sample_ids], 
	                                            (mat_phase_data.shape[1], 1)).T
	pat_match = pat_phase_data == (2+np.tile([sample_to_affected[x]=='2' for x in sample_ids], 
	                                            (pat_phase_data.shape[1], 1)).T)
	sample_id_to_index = dict([(x, i) for i, x in enumerate(sample_ids)])

	male_simplex_familysizes = defaultdict(int)
	female_simplex_familysizes = defaultdict(int)
	multiplex_familysizes = defaultdict(int)
	for family_key, inds in family_to_inds.items():
		child_indices = [sample_id_to_index[x] for x in inds[2:]]
		is_multiplex = sum([sample_to_affected[x]=='2' for x in inds[2:]]) > 1
		is_male_simplex = False
		if not is_multiplex:
			is_male_simplex = [sample_to_sex[x]=='1' for x in inds[2:] if sample_to_affected[x]=='2'][0]
	    
		no_missing = np.all(mat_phase_data[child_indices, :]>=0, axis=0)
		family_mat_match = np.sum(mat_match[child_indices, :], axis=0)
		max_family_mat_match = np.maximum(family_mat_match, len(inds[2:]) - family_mat_match)
		mat_st[len(inds[2:])*np.ones((np.sum(no_missing),), dtype=int), 
			   (2*is_multiplex+is_male_simplex)*np.ones((np.sum(no_missing),), dtype=int),
			   max_family_mat_match[no_missing], 
			   np.where(no_missing)[0]] += 1

		no_missing = np.all(pat_phase_data[child_indices, :]>=0, axis=0)
		family_pat_match = np.sum(pat_match[child_indices, :], axis=0)
		max_family_pat_match = np.maximum(family_pat_match, len(child_indices) - family_pat_match)
		pat_st[len(inds[2:])*np.ones((np.sum(no_missing),), dtype=int), 
			   (2*is_multiplex+is_male_simplex)*np.ones((np.sum(no_missing),), dtype=int),
			   max_family_pat_match[no_missing], 
			   np.where(no_missing)[0]] += 1

		if is_multiplex:
			multiplex_familysizes[len(inds)] += 1
		elif is_male_simplex:
			male_simplex_familysizes[len(inds)] += 1
		else:
			female_simplex_familysizes[len(inds)] += 1

	assert np.all(np.sum(mat_st, axis=(0, 1, 2))) <= len(family_to_inds)
	assert np.all(np.sum(pat_st, axis=(0, 1, 2))) <= len(family_to_inds)

	print('male simplex family sizes', male_simplex_familysizes)
	print('female simplex family sizes', female_simplex_familysizes)
	print('multiplex family sizes', multiplex_familysizes)
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
	if np.sum(indices) > 0:
		pvalue = scipy.stats.chisquare(obs[indices], exp[indices], ddof=np.sum(num_fams>0)-1).pvalue	
	else:
		pvalue = 1

	return pvalue

def calculate_pvalues(mat_st, pat_st):
	# pos, simplexF/simplexM/simplex/mutiplex/all, mat/pat/combined
	pvalues = np.ones((mat_st.shape[3], 5, 3))
	for index in range(mat_st.shape[3]):
		# mat
		pvalues[index, 0, 0] = calculate_pvalue(mat_st[:, 0, :, index])
		pvalues[index, 1, 0] = calculate_pvalue(mat_st[:, 1, :, index])
		pvalues[index, 2, 0] = calculate_pvalue(mat_st[:, 0, :, index]+mat_st[:, 1, :, index])
		pvalues[index, 3, 0] = calculate_pvalue(mat_st[:, 2, :, index])
		pvalues[index, 4, 0] = calculate_pvalue(np.sum(mat_st[:, :, :, index], axis=1))

		# pat
		pvalues[index, 0, 1] = calculate_pvalue(pat_st[:, 0, :, index])
		pvalues[index, 1, 1] = calculate_pvalue(pat_st[:, 1, :, index])
		pvalues[index, 2, 1] = calculate_pvalue(pat_st[:, 0, :, index]+pat_st[:, 1, :, index])
		pvalues[index, 3, 1] = calculate_pvalue(pat_st[:, 2, :, index])
		pvalues[index, 4, 1] = calculate_pvalue(np.sum(pat_st[:, :, :, index], axis=1))
	    
		# both
		pvalues[index, 0, 2] = calculate_pvalue(mat_st[:, 0, :, index]+pat_st[:, 0, :, index])
		pvalues[index, 1, 2] = calculate_pvalue(mat_st[:, 1, :, index]+pat_st[:, 1, :, index])
		pvalues[index, 2, 2] = calculate_pvalue(mat_st[:, 0, :, index]+pat_st[:, 0, :, index]+mat_st[:, 1, :, index]+pat_st[:, 1, :, index])
		pvalues[index, 3, 2] = calculate_pvalue(mat_st[:, 2, :, index]+pat_st[:, 2, :, index])
		pvalues[index, 4, 2] = calculate_pvalue(np.sum(mat_st[:, :, :, index], axis=1)+np.sum(pat_st[:, :, :, index], axis=1))
	return pvalues


if __name__ == "__main__":
	sample_to_affected, sample_to_sex = pull_phenotype(ped_file)

	sample_ids, family_to_inds = pull_samples(phase_dir, chrom)
	print('samples', len(sample_ids))
	print('families', len(family_to_inds))

	regions, mat_phase_data, pat_phase_data = pull_phase(phase_dir, chrom, sample_ids, family_to_inds, interval)
	print('regions', regions.shape[0])
	print(np.unique(mat_phase_data, return_counts=True), np.unique(pat_phase_data, return_counts=True))

	interval_positions = np.arange(0, regions[-1, 1], interval)
	intervals = np.hstack((interval_positions[:-1, np.newaxis], interval_positions[1:, np.newaxis]))
	intervals[-1, 1] = regions[-1, 1]
	print('intervals', intervals.shape[0])

	mat_phase_data_interval, pat_phase_data_interval = reduce_to_intervals(intervals, mat_phase_data, pat_phase_data, regions)
	print(np.unique(mat_phase_data_interval, return_counts=True), np.unique(pat_phase_data_interval, return_counts=True))

	mat_st, pat_st = generate_test_statistic(mat_phase_data_interval, pat_phase_data_interval, sample_ids, sample_to_affected, sample_to_sex)
	print('IST statistic computed')

	pvalues = calculate_pvalues(mat_st, pat_st)
	print('pvalues computed')

	print(pvalues.shape)
	np.save(outfile, pvalues)
	np.save(outfile + '.regions', intervals)
	print('results saved to %s' % outfile)


