from collections import defaultdict, namedtuple, Counter
from itertools import combinations
import numpy as np
import scipy.stats
import sys
from os import listdir

#chrom = sys.argv[1]

##phase_dir = 'phased_spark'
#ped_file = 'data/spark.ped'
#family_sizes = [4, 5, 6, 7]
#identicals_file = 'sibpair_similarity/spark_identicals.txt'


#phase_dir = 'phased_ssc'
#ped_file = 'data/ssc.ped'
#family_sizes = [4]
#identicals_file = 'sibpair_similarity/ssc_identicals.txt'

#phase_dir = 'phased_ihart'
#ped_file = 'data/v34.vcf.ped'
#family_sizes = [4, 5, 6]
#identicals_file = 'sibpair_similarity/ihart_identicals.txt'

#phase_dir = 'phased_ancestry'
#ped_file = 'data/ancestry.ped'
#family_sizes = [4, 5, 6, 7]

phase_dir = 'phased_spark_quads'
ped_file = 'data/spark.ped.quads.ped'
identicals_file = 'sibpair_similarity/spark_quads_identicals.txt'
build = '38'

#phase_dir = 'phased_ancestry_quads'
#phenotype_file = '../DATA/ancestry/ancestryDNA.ped.quads.ped'
#identicals_file = 'sibpair_similarity/ssc_identicals.txt'

#phase_dir = 'phased_ihart_quads'
#phenotype_file = 'data/v34.vcf.ped'
#identicals_file = 'sibpair_similarity/ihart_identicals.txt'

chroms = [str(x) for x in range(1, 23)]
#chroms = ['10']
#interval = 5000000
#name = str(interval)
name = 'cyto'


def pull_phenotype_ped(ped_file):
	sample_to_sex = dict()
	sample_to_affected = dict()
	with open(ped_file, 'r') as f:
		for line in f:
			pieces = line.strip().split('\t')
			sample_to_sex[pieces[1]] = pieces[4]
			sample_to_affected[pieces[1]] = pieces[5]
	return sample_to_affected, sample_to_sex

Sibpair = namedtuple('Sibpair', ['family', 'sibling1', 'sibling2', 'mom', 'dad', 'num_affected', 'num_males'])
def pull_sibpairs(phase_dir, identicals_file, sample_to_affected, sample_to_sex):

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

	assert len(sibpairs) == len(set(sibpairs)) # should have no duplicates
	return family_to_inds, sibpairs

def pull_sibpair_positions(phase_dir, chrom):
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

	return positions
	

def pull_sibpair_recombination(phase_dir, chrom, sibpairs, family_to_inds, positions):
	sibpair_to_index = dict([((x.family, x.sibling1, x.sibling2), i) for i, x in enumerate(sibpairs)])

	position_to_index = dict([(x, i) for i, x in enumerate(positions)])
	regions = np.hstack((positions[:-1, np.newaxis], positions[1:, np.newaxis]))
	region_lengths = regions[:, 1]-regions[:, 0]
	print(regions.shape)

	# pull phase data
	# sibpair, position
	mat_recombination = np.zeros((len(sibpair_to_index), regions.shape[0]))
	pat_recombination = np.zeros((len(sibpair_to_index), regions.shape[0]))
	for filename in listdir(phase_dir):
		if filename.endswith('.phased.txt'):
			family_key = filename[:-11]
			inds = family_to_inds[family_key]
			if len(inds)==4 and (family_key, inds[2], inds[3]) in sibpair_to_index:
				sibpair_index = sibpair_to_index[(family_key, inds[2], inds[3])]

				with open('%s/%s' % (phase_dir, filename), 'r')  as f:
					next(f) # skip header

					states = []
					positions = []
					for line in f:
						pieces = line.strip().split('\t')
						if pieces[0][3:] == chrom:
							states.append([int(x) for x in pieces[1:-2]])
							positions.append([int(x) for x in pieces[-2:]])

					states = np.array(states).T
					positions = np.array(positions).T

					# mat
					last_known_state = None
					last_known_pos = None
					for s, start_pos, end_pos in zip(states[10, :], positions[0, :], positions[1, :]):
						if s == -1:
							pass
						elif s == last_known_state:
							last_known_pos = end_pos
						else:
							if last_known_state is not None:
								recomb_start_pos, recomb_end_pos = last_known_pos, start_pos
								pos_indices = np.arange(position_to_index[recomb_start_pos], position_to_index[recomb_end_pos])
								mat_recombination[sibpair_index, pos_indices] = region_lengths[pos_indices]/(recomb_end_pos-recomb_start_pos)
							last_known_state = s
							last_known_pos = end_pos

					# pat
					last_known_state = None
					last_known_pos = None
					for s, start_pos, end_pos in zip(states[11, :], positions[0, :], positions[1, :]):
						if s == -1:
							pass
						elif s == last_known_state:
							last_known_pos = end_pos
						else:
							if last_known_state is not None:
								recomb_start_pos, recomb_end_pos = last_known_pos, start_pos
								pos_indices = np.arange(position_to_index[recomb_start_pos], position_to_index[recomb_end_pos])
								pat_recombination[sibpair_index, pos_indices] = region_lengths[pos_indices]/(recomb_end_pos-recomb_start_pos)
							last_known_state = s
							last_known_pos = end_pos

	return regions, mat_recombination, pat_recombination

def reduce_to_intervals(intervals, mat_recombination, pat_recombination, regions):
	assert mat_recombination.shape == pat_recombination.shape
	mat_recombination_interval = np.zeros((mat_recombination.shape[0], intervals.shape[0]))
	pat_recombination_interval = np.zeros((pat_recombination.shape[0], intervals.shape[0]))

	for i, (interval_start, interval_end) in enumerate(intervals):

		overlap = np.minimum(regions[:, 1], interval_end) - np.maximum(regions[:, 0], interval_start)
		assert np.all((overlap<= 0) | (overlap == (regions[:, 1]-regions[:, 0])) )
		
		for region_index in np.where(overlap>0)[0]:
			mat_recombination_interval[:, i] += mat_recombination[:, region_index]
			pat_recombination_interval[:, i] += pat_recombination[:, region_index]

	return mat_recombination_interval, pat_recombination_interval

def generate_test_statistic(mat_recombination, pat_recombination, sibpairs):

	#family_type_to_index = dict([('male_simplex', 0), ('female_simplex', 1), ('multiplex', 2)])

	# positions, UU/AU/AA, mat/pat, no recomb/recomb
	r = np.zeros((mat_recombination.shape[1], 3, 2, 2))

	for sibpair_index, sibpair in enumerate(sibpairs):
		#if sibpair.family_type in family_type_to_index:
		#family_type_index = family_type_to_index[sibpair.family_type]

		r[:, sibpair.num_affected, 0, 0] += (1-mat_recombination[sibpair_index, :])
		r[:, sibpair.num_affected, 0, 1] += mat_recombination[sibpair_index, :]

		r[:, sibpair.num_affected, 1, 0] += (1-pat_recombination[sibpair_index, :])
		r[:, sibpair.num_affected, 1, 1] += pat_recombination[sibpair_index, :]


	return r

def calculate_pvalue(obs):
	#obs: UU/AU/AA, no recomb/recomb

	n = 2*np.sum(obs, axis=1)

	ps = np.ones((3,))

	if n[0] != 0 and n[2] != 0:
		try:
			p1 = obs[0, 1]/n[0]
			p2 = obs[2, 1]/n[2]
			p = (obs[0, 1] + obs[2, 1])/(n[0]+n[2])
			z = (p1-p2)/np.sqrt(p*(1-p)*((1/n[0])+(1/n[2])))
			ps[0] = scipy.stats.norm.cdf(z)
		except:
			pass

	if n[0] != 0 and n[1] != 0:
		try:
			p1 = obs[0, 1]/n[0]
			p2 = obs[1, 1]/n[1]
			p = (obs[0, 1] + obs[1, 1])/(n[0]+n[1])
			z = (p1-p2)/np.sqrt(p*(1-p)*((1/n[0])+(1/n[1])))
			ps[1] = scipy.stats.norm.cdf(z)
		except:
			pass

	if n[1] != 0 and n[2] != 0:
		try:
			p1 = obs[1, 1]/n[1]
			p2 = obs[2, 1]/n[2]
			p = (obs[1, 1] + obs[2, 1])/(n[1]+n[2])
			z = (p1-p2)/np.sqrt(p*(1-p)*((1/n[1])+(1/n[2])))
			ps[2] = scipy.stats.norm.cdf(z)
		except:
			pass

	return ps

def calculate_pvalues(r):
	# positions, UU/AU/AA, mat/pat, no recomb/recomb

	# pos, mat/pat/both
	pvalues = np.ones((r.shape[0], 3, 3))
	for index in range(r.shape[0]):
		pvalues[index, 0, :] = calculate_pvalue(r[index, :, 0, :])
		pvalues[index, 1, :] = calculate_pvalue(r[index, :, 1, :])
		pvalues[index, 2, :] = calculate_pvalue(np.sum(r[index, :, :, :], axis=1))

	return pvalues


if __name__ == "__main__":

	sample_to_affected, sample_to_sex = pull_phenotype_ped(ped_file)

	family_to_inds, sibpairs = pull_sibpairs(phase_dir, identicals_file, sample_to_affected, sample_to_sex)
	print('sibpairs', len(sibpairs))
	print('families', len(family_to_inds))
	
	for chrom in chroms:
		outfile = '%s/chr.%s.recomb.pvalues.intervals.%s' % (phase_dir, chrom, name)

		positions = pull_sibpair_positions(phase_dir, chrom)

		#intervals = np.arange(0, max(positions), interval)

		intervals = set()
		with open('data/cytoBand%s.txt' % build, 'r') as f:
			for line in f:
				pieces = line.strip().split('\t')
				if pieces[0].startswith('chr') and pieces[0][3:] == chrom:
					start_pos, end_pos = int(pieces[1]), int(pieces[2])
					intervals.update([
						start_pos, 
						0.75*start_pos + 0.25*end_pos,
						0.5*start_pos + 0.5*end_pos,
						0.25*start_pos + 0.75*end_pos,
						end_pos
						])
		intervals.add(max(positions))
		intervals = np.array(sorted(intervals))

		positions.update(intervals)
		positions = np.array(sorted(positions))

		regions, mat_recombination, pat_recombination = pull_sibpair_recombination(phase_dir, chrom, sibpairs, family_to_inds, positions)
		print('regions', regions.shape[0])

		intervals = np.hstack((intervals[:-1, np.newaxis], intervals[1:, np.newaxis]))
		intervals[-1, 1] = regions[-1, 1]
		print('intervals', intervals.shape[0])

		mat_recombination_interval, pat_recombination_interval = reduce_to_intervals(intervals, mat_recombination, pat_recombination, regions)
		print(mat_recombination_interval.shape)

		r = generate_test_statistic(mat_recombination_interval, pat_recombination_interval, sibpairs)
		np.save(outfile + '.contingency', r)
		print('statistic computed')

		pvalues = calculate_pvalues(r)
		print('pvalues computed')

		print(pvalues.shape)
		np.save(outfile, pvalues)
		np.save(outfile + '.regions', intervals)
		print('results saved to %s' % outfile)


