from collections import defaultdict, namedtuple, Counter
from itertools import combinations
import numpy as np
import scipy.stats
import sys
from os import listdir


phase_dir = 'phased_spark_quads_consang'
ped_file = 'data/spark.ped.quads.ped'
identicals_file = 'sibpair_similarity/spark_quads_identicals.txt'
phenotype = ''
##phenotype = 'q10_hand_tool'

chroms = [str(x) for x in range(1, 23)]
#chroms = ['10']
interval = 500000

def pull_ped(ped_file):
	sample_to_sex = dict()
	sample_to_affected = dict()
	with open(ped_file, 'r') as f:
		for line in f:
			pieces = line.strip().split('\t')
			sample_to_sex[pieces[1]] = pieces[4]
			sample_to_affected[pieces[1]] = pieces[5]
	return sample_to_affected, sample_to_sex


Sibpair = namedtuple('Sibpair', ['family', 'sibling1', 'sibling2', 'mom', 'dad', 'num_affected', 'num_males'])
def pull_sibpairs(phase_dir, identicals_file, chrom, sample_to_affected, sample_to_sex):

	# pull identicals
	leave_out = set()
	with open(identicals_file, 'r') as f:
		for line in f:
			pieces = line.strip().split('\t')
			leave_out.update(pieces[1:])

	# pull individuals
	family_to_inds = defaultdict(list)
	sibpairs = list()
	for filename in listdir(phase_dir):
		if filename.endswith('.phased.txt'):
			family_key = filename[:-11]
			with open('%s/%s' % (phase_dir, filename), 'r')  as f:
				header = next(f).strip().split('\t')
				# check that we have a typical nuclear family structure
				if tuple(header[1:5]) == ('m1_del', 'm2_del', 'p1_del', 'p2_del'):
					individuals = [header[i][:-4] for i in range(5, len(header)-3, 2)]
					family_to_inds[family_key] = individuals
					for child1, child2 in combinations(individuals[2:], 2):
						if child1 not in leave_out and child2 not in leave_out and child1 in sample_to_affected and child2 in sample_to_affected:
							sibpairs.append(Sibpair(family_key, child1, child2, individuals[0], individuals[1], 
								int(sample_to_affected[child1]=='2')+int(sample_to_affected[child2]=='2'),
								int(sample_to_sex[child1]=='1')+int(sample_to_sex[child2]=='1')))

	sibpairs = sorted(sibpairs)

	print('num_affected', Counter([x.num_affected for x in sibpairs]))
	print('num_males', Counter([x.num_males for x in sibpairs]))
	print('num_affected/num_males', Counter([(x.num_affected, x.num_males) for x in sibpairs]))

	assert len(sibpairs) == len(set(sibpairs)) # should have no duplicates
	return family_to_inds, sibpairs

def pull_sibpair_matches(phase_dir, chrom, sibpairs, family_to_inds, interval):
	sibpair_to_index = dict([((x.family, x.sibling1, x.sibling2), i) for i, x in enumerate(sibpairs)])

	# pull positions
	positions = set()
	for filename in listdir(phase_dir):
		if filename.endswith('.phased.txt'):
			with open('%s/%s' % (phase_dir, filename), 'r')  as f:
				next(f) # skip header

				for line in f:
					pieces = line.strip().split('\t')
					if pieces[0][3:] == chrom:
						start_pos, end_pos = [int(x) for x in pieces[-2:]]
						assert end_pos >= start_pos

						positions.add(start_pos)
						positions.add(end_pos)

	positions.update(np.arange(0, max(positions), interval))
	positions = np.array(sorted(positions))
	position_to_index = dict([(x, i) for i, x in enumerate(positions)])
	regions = np.hstack((positions[:-1, np.newaxis], positions[1:, np.newaxis]))
	print(regions.shape)

	# pull phase data
	# sibpair, position
	mat_match_data = -np.ones((len(sibpair_to_index), regions.shape[0]), dtype=int)
	pat_match_data = -np.ones((len(sibpair_to_index), regions.shape[0]), dtype=int)
	for filename in listdir(phase_dir):
		if filename.endswith('.phased.txt'):
			family_key = filename[:-11]
			with open('%s/%s' % (phase_dir, filename), 'r')  as f:
				next(f) # skip header

				for line in f:
					pieces = line.strip().split('\t')
					if pieces[0][3:] == chrom:
						start_pos, end_pos = [int(x) for x in pieces[-2:]]
						state = np.array([int(x) for x in pieces[1:-2]])

						inds = family_to_inds[family_key]

						sibpair_indices = np.array([sibpair_to_index[(family_key, child1, child2)] for child1, child2 in combinations(inds[2:], 2) if (family_key, child1, child2) in sibpair_to_index])
						if len(sibpair_indices) > 0:
							pos_indices = np.arange(position_to_index[start_pos], position_to_index[end_pos])

							mat_phase = np.array(list(combinations(state[np.arange(8, 4+2*len(inds), 2)], 2)))
							no_missing = np.all(mat_phase>=0, axis=1)
							for sibpair_index in sibpair_indices[no_missing & (mat_phase[:, 0]==mat_phase[:, 1])]:
								mat_match_data[sibpair_index, pos_indices] = 1
							for sibpair_index in sibpair_indices[no_missing & (mat_phase[:, 0]!=mat_phase[:, 1])]:
								mat_match_data[sibpair_index, pos_indices] = 0
							
							pat_phase = np.array(list(combinations(state[np.arange(9, 4+2*len(inds), 2)], 2)))
							no_missing = np.all(pat_phase>=0, axis=1)
							for sibpair_index in sibpair_indices[no_missing & (pat_phase[:, 0]==pat_phase[:, 1])]:
								pat_match_data[sibpair_index, pos_indices] = 1
							for sibpair_index in sibpair_indices[no_missing & (pat_phase[:, 0]!=pat_phase[:, 1])]:
								pat_match_data[sibpair_index, pos_indices] = 0
				
	return regions, mat_match_data, pat_match_data

def reduce_to_intervals(intervals, mat_match_data, pat_match_data, regions):
	assert mat_match_data.shape == pat_match_data.shape
	mat_match_data_interval = -np.ones((mat_match_data.shape[0], intervals.shape[0]), dtype=int)
	pat_match_data_interval = -np.ones((pat_match_data.shape[0], intervals.shape[0]), dtype=int)

	region_length = regions[:, 1]-regions[:, 0]

	for i, (interval_start, interval_end) in enumerate(intervals):
		overlap = np.minimum(regions[:, 1], interval_end) - np.maximum(regions[:, 0], interval_start)
		assert np.all((overlap<= 0) | (overlap == region_length))

		is_overlapped = overlap>0

		is_mat_matched = np.sum(region_length[is_overlapped] * (mat_match_data[:, is_overlapped]==1), axis=1) > (interval_end-interval_start)*0.9
		mat_match_data_interval[is_mat_matched, i] = 1

		is_mat_mismatched = np.sum(region_length[is_overlapped] * (mat_match_data[:, is_overlapped]==0), axis=1) > (interval_end-interval_start)*0.9
		mat_match_data_interval[is_mat_mismatched, i] = 0

		is_pat_matched = np.sum(region_length[is_overlapped] * (pat_match_data[:, is_overlapped]==1), axis=1) > (interval_end-interval_start)*0.9
		pat_match_data_interval[is_pat_matched, i] = 1

		is_pat_mismatched = np.sum(region_length[is_overlapped] * (pat_match_data[:, is_overlapped]==0), axis=1) > (interval_end-interval_start)*0.9
		pat_match_data_interval[is_pat_mismatched, i] = 0


	return mat_match_data_interval, pat_match_data_interval

def generate_test_statistic(mat_match_data, pat_match_data, sibpairs):

	#family_type_to_index = dict([('male_simplex', 0), ('female_simplex', 1), ('multiplex', 2)])

	# positions, FF/MF/MM, UU/AU/AA, num match
	r = np.zeros((mat_match_data.shape[1], 3, 3, 3), dtype=int)

	# positions, FF/MF/MM, UU/AU/AA, num match
	r_mat = np.zeros((mat_match_data.shape[1], 3, 3, 2), dtype=int)

	# positions, FF/MF/MM, UU/AU/AA, num match
	r_pat = np.zeros((mat_match_data.shape[1], 3, 3, 2), dtype=int)

	for sibpair_index, sibpair in enumerate(sibpairs):
		#if sibpair.family_type in family_type_to_index:
		#family_type_index = family_type_to_index[sibpair.family_type]

		no_missing_mat = mat_match_data[sibpair_index, :]>=0
		no_missing_pat = pat_match_data[sibpair_index, :]>=0
		no_missing_all = no_missing_mat & no_missing_pat

		r[np.where(no_missing_all)[0],
		  sibpair.num_males, 
		  sibpair.num_affected, 
		  mat_match_data[sibpair_index, no_missing_all]+pat_match_data[sibpair_index, no_missing_all], 
		] += 1

		r_mat[np.where(no_missing_mat)[0],
		  sibpair.num_males, 
		  sibpair.num_affected, 
		  mat_match_data[sibpair_index, no_missing_mat], 
		] += 1

		r_pat[np.where(no_missing_pat)[0],
		  sibpair.num_males, 
		  sibpair.num_affected, 
		  pat_match_data[sibpair_index, no_missing_pat], 
		] += 1
	return r, r_mat, r_pat

def calculate_pvalue(obs):
	#obs: UU/AU/AA, num match

	n = np.sum(obs, axis=1)

	p0, p1, p2, p3, p4, p5, p6 = 1, 1, 1, 1, 1, 1, 1

	if n[0] != 0:
		p0 = scipy.stats.binom_test(obs[0, 1] + 2*obs[0, 2], 2*n[0], p=0.5, alternative='greater')

	if n[1] != 0:
		p1 = scipy.stats.binom_test(obs[1, 1] + 2*obs[1, 2], 2*n[1], p=0.5, alternative='less')

	if n[2] != 0:
		p2 = scipy.stats.binom_test(obs[2, 1] + 2*obs[2, 2], 2*n[2], p=0.5, alternative='greater')

	if n[1] != 0 and n[2] != 0:
		try:
			p3 = scipy.stats.chi2_contingency([[obs[1, 1] + 2*obs[1, 2], obs[1, 1] + 2*obs[1, 0]],
											   [obs[2, 1] + 2*obs[2, 2], obs[2, 1] + 2*obs[2, 0]]])[1]
		except:
			pass

	if n[1] != 0 and n[0] != 0:
		try:
			p4 = scipy.stats.chi2_contingency([[obs[1, 1] + 2*obs[1, 2], obs[1, 1] + 2*obs[1, 0]],
											   [obs[0, 1] + 2*obs[0, 2], obs[0, 1] + 2*obs[0, 0]]])[1]
		except:
			pass

	if n[1] != 0 and (n[0]+n[2]) != 0:
		try:
			p5 = scipy.stats.chi2_contingency([[obs[1, 1] + 2*obs[1, 2], obs[1, 1] + 2*obs[1, 0]],
											   [obs[0, 1] + 2*obs[0, 2]+obs[2, 1] + 2*obs[2, 2], obs[0, 1] + 2*obs[0, 0]+obs[2, 1] + 2*obs[2, 0]]])[1]
		except:
			pass

	if (n[0]+n[2]) != 0:
		p6 = scipy.stats.binom_test(obs[0, 1] + 2*obs[0, 2] + obs[2, 1] + 2*obs[2, 2], 
									(2*n[0])+(2*n[2]), p=0.5, alternative='greater')

	return p0, p1, p2, p3, p4, p5, p6

def calculate_pvalue_mat_pat(obs):
	#obs: UU/AU/AA, num match

	n = np.sum(obs, axis=1)
	#p = (obs.T/n).T

	p0, p1, p2, p3, p4, p5, p6 = 1, 1, 1, 1, 1, 1, 1

	if n[0] != 0:
		p0 = scipy.stats.binom_test(obs[0, 1], n[0], p=0.5, alternative='greater')

	if n[1] != 0:
		p1 = scipy.stats.binom_test(obs[1, 1], n[1], p=0.5, alternative='less')

	if n[2] != 0:
		p2 = scipy.stats.binom_test(obs[2, 1], n[2], p=0.5, alternative='greater')

	if n[1] != 0 and n[2] != 0:
		try:
			p3 = scipy.stats.chi2_contingency([[obs[1, 1], n[1]-obs[1, 1]],
											 [obs[2, 1], n[2]-obs[2, 1]]])[1]
		except:
			pass

	if n[1] != 0 and n[0] != 0:
		try:
			p4 = scipy.stats.chi2_contingency([[obs[1, 1], n[1]-obs[1, 1]],
											 [obs[0, 1], n[0]-obs[0, 1]]])[1]
		except:
			pass

	if n[1] != 0 and (n[0]+n[2]) != 0:
		try:
			p5 = scipy.stats.chi2_contingency([[obs[1, 1], n[1]-obs[1, 1]],
											 [obs[0, 1]+obs[2, 1], n[0]+n[2]-obs[0, 1]-obs[2, 1]]])[1]
		except:
			pass

	if (n[0]+n[2]) != 0:
		p6 = scipy.stats.binom_test(obs[0, 1] + obs[2, 1], 
									n[0]+n[2], p=0.5, alternative='greater')

	return p0, p1, p2, p3, p4, p5, p6

def calculate_pvalues(r, r_mat, r_pat):
	# r: positions, FF/MF/MM, UU/AU/AA, num match

	# pos, FF/MF/MM/all, mat/pat/both, UU/AU/AA/AUvsAA/AUvsUU/AUvsAAUU/all
	pvalues = np.ones((r.shape[0], 4, 3, 7))
	for index in range(r.shape[0]):
		# both
		pvalues[index, 0, 2, :] = calculate_pvalue(r[index, 0, :, :])
		pvalues[index, 1, 2, :] = calculate_pvalue(r[index, 1, :, :])
		pvalues[index, 2, 2, :] = calculate_pvalue(r[index, 2, :, :])
		pvalues[index, 3, 2, :] = calculate_pvalue(np.sum(r[index, :, :, :], axis=0))

		# mat
		pvalues[index, 0, 0, :] = calculate_pvalue_mat_pat(r_mat[index, 0, :, :])
		pvalues[index, 1, 0, :] = calculate_pvalue_mat_pat(r_mat[index, 1, :, :])
		pvalues[index, 2, 0, :] = calculate_pvalue_mat_pat(r_mat[index, 2, :, :])
		pvalues[index, 3, 0, :] = calculate_pvalue_mat_pat(np.sum(r_mat[index, :, :, :], axis=0))

		# pat
		pvalues[index, 0, 1, :] = calculate_pvalue_mat_pat(r_pat[index, 0, :, :])
		pvalues[index, 1, 1, :] = calculate_pvalue_mat_pat(r_pat[index, 1, :, :])
		pvalues[index, 2, 1, :] = calculate_pvalue_mat_pat(r_pat[index, 2, :, :])
		pvalues[index, 3, 1, :] = calculate_pvalue_mat_pat(np.sum(r_pat[index, :, :, :], axis=0))


	return pvalues


if __name__ == "__main__":

	sample_to_affected, sample_to_sex = pull_ped(ped_file)
	
	for chrom in chroms:
		outfile = '%s/chr.%s.IST.pvalues.be.%sintervals.%d' % (phase_dir, chrom, phenotype, interval)

		family_to_inds, sibpairs = pull_sibpairs(phase_dir, identicals_file, chrom, sample_to_affected, sample_to_sex)
		print('sibpairs', len(sibpairs))
		print('families', len(family_to_inds))

		regions, mat_match_data, pat_match_data = pull_sibpair_matches(phase_dir, chrom, sibpairs, family_to_inds, interval)
		print('regions', regions.shape[0])
		print(np.unique(mat_match_data, return_counts=True), np.unique(pat_match_data, return_counts=True))

		interval_positions = np.arange(0, regions[-1, 1], interval)
		intervals = np.hstack((interval_positions[:-1, np.newaxis], interval_positions[1:, np.newaxis]))
		intervals[-1, 1] = regions[-1, 1]
		print('intervals', intervals.shape[0])

		mat_match_data_interval, pat_match_data_interval = reduce_to_intervals(intervals, mat_match_data, pat_match_data, regions)
		print(mat_match_data_interval.shape)
		print(np.unique(mat_match_data_interval, return_counts=True), np.unique(pat_match_data_interval, return_counts=True))

		r, r_mat, r_pat = generate_test_statistic(mat_match_data_interval, pat_match_data_interval, sibpairs)
		np.save(outfile + '.contingency', np.sum(r, axis=1))
		print('statistic computed')

		pvalues = calculate_pvalues(r, r_mat, r_pat)
		print('pvalues computed')

		print(pvalues.shape)
		np.save(outfile, pvalues)
		np.save(outfile + '.regions', intervals)
		print('results saved to %s' % outfile)


