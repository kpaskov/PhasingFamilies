from collections import defaultdict, namedtuple, Counter
from itertools import combinations
import numpy as np
import scipy.stats
import sys
from os import listdir
import json

#phase_dirs = ['phased_ihart.ms2_quads']
#ped_files = ['../DATA/ihart.ms2/ihart.ped.quads.ped']
#identicals_files = ['sibpair_similarity/ihart.ms2_quads_identicals.txt']
#out_dir = 'phased_ihart.ms2_quads'
#build = '38'

#phase_dirs = ['phased_ssc.hg38']
#ped_files = ['../DATA/ssc.hg38/ssc.ped']
#identicals_files = ['sibpair_similarity/ssc.hg38_identicals.txt']
#out_dir = 'phased_ssc.hg38'
#build = '38'

#phase_dirs = ['phased_mssng_quads']
#ped_files = ['../DATA/mssng/mssng.ped.quads.ped']
#identicals_files = ['sibpair_similarity/mssng_quads_identicals.txt']
#out_dir = 'phased_mssng_quads'
#build = '38'

phase_dirs = ['phased_spark_quads']
ped_files = ['../DATA/spark/spark.ped.quads.ped']
identicals_files = ['sibpair_similarity/spark_quads_identicals.txt']
out_dir = 'phased_spark_quads'
build = '38'
save_contingency = True

#phase_dirs = ['phased_ancestry_quads']
#ped_files = ['../DATA/ancestry/ancestry.ped.quads.ped']
#identicals_files = ['sibpair_similarity/ancestry_quads_identicals.txt']
#out_dir = 'phased_ancestry_quads'
#build = '37'
#save_contingency = True

#phase_dirs = [
#'phased_ihart.ms2_quads', 
#'phased_ssc.hg38', 
#'phased_mssng_quads', 
#]
#ped_files = [
#'../DATA/ihart.ms2/ihart.ped.quads.ped', 
#'../DATA/ssc.hg38/ssc.ped',
#'../DATA/mssng/mssng.ped.quads.ped',
#]
#identicals_files = [
#'sibpair_similarity/ihart.ms2_quads_identicals.txt',
#'sibpair_similarity/ssc.hg38_identicals.txt',
#'sibpair_similarity/mssng_quads_identicals.txt',
#]
#out_dir = 'sibpair_wgs'


chroms = [str(x) for x in range(1, 23)]
#chroms = ['10']
interval = 500000

def pull_phenotype_ped(ped_file):
	sample_to_sex = dict()
	sample_to_affected = dict()
	with open(ped_file, 'r') as f:
		for line in f:
			pieces = line.strip().split('\t')
			sample_to_sex[pieces[1]] = pieces[4]
			sample_to_affected[pieces[1]] = pieces[5]
	return sample_to_affected, sample_to_sex

Sibpair = namedtuple('Sibpair', ['family', 'sibling1', 'sibling2', 'mom', 'dad', 'phase_dir', 'num_affected', 'num_males'])
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
								phase_dir,
								int(sample_to_affected[child1]=='2')+int(sample_to_affected[child2]=='2'),
								int(sample_to_sex[child1]=='1')+int(sample_to_sex[child2]=='1')))

	sibpairs = sorted(sibpairs)

	assert len(sibpairs) == len(set(sibpairs)) # should have no duplicates
	return family_to_inds, sibpairs

def pull_sibpair_matches(phase_dirs, chrom, sibpairs, family_to_inds, interval_bins):
	sibpair_to_index = dict([((x.family, x.sibling1, x.sibling2), i) for i, x in enumerate(sibpairs)])

	# pull phase data
	# sibpair, interval, nomatch/match
	is_mat_match = -np.ones((len(sibpair_to_index), len(interval_bins)-1), dtype=int)
	is_pat_match = -np.ones((len(sibpair_to_index), len(interval_bins)-1), dtype=int)

	interval_lengths = interval_bins[1:]-interval_bins[:-1]

	for sibpair_index, sibpair in enumerate(sibpairs):
		mat_match_data = np.zeros((len(interval_bins)-1, 2), dtype=int)
		pat_match_data = np.zeros((len(interval_bins)-1, 2), dtype=int)
		with open('%s/%s.phased.txt' % (sibpair.phase_dir, sibpair.family), 'r')  as f:
			next(f) # skip header

			for line in f:
				pieces = line.strip().split('\t')
				if pieces[0][3:] == chrom:
					start_pos, end_pos = [int(x) for x in pieces[-2:]]
					state = np.array([int(x) for x in pieces[1:-2]])

					inds = family_to_inds[sibpair.family]
					sib1_ind_index, sib2_ind_index = inds.index(sibpair.sibling1), inds.index(sibpair.sibling2)
					sib1_mat_index, sib2_mat_index = 4+2*sib1_ind_index, 4+2*sib2_ind_index
					sib1_pat_index, sib2_pat_index = 5+2*sib1_ind_index, 5+2*sib2_ind_index

					start_index, end_index = np.searchsorted(interval_bins, [start_pos, end_pos])

					for index in range(start_index, end_index):
						overlap = min(interval_bins[index], end_pos) - max(interval_bins[index-1], start_pos)
						assert overlap >= 0
						if (state[sib1_mat_index] != -1) and (state[sib2_mat_index] != -1):
							mat_match_data[index-1, int(state[sib1_mat_index]==state[sib2_mat_index])] += overlap

						if (state[sib1_pat_index] != -1) and (state[sib2_pat_index] != -1):
							pat_match_data[index-1, int(state[sib1_pat_index]==state[sib2_pat_index])] += overlap

		assert np.all(np.sum(mat_match_data, axis=1) <= interval_lengths)
		assert np.all(np.sum(pat_match_data, axis=1) <= interval_lengths)

		is_mat_match[sibpair_index, mat_match_data[:, 0]>=0.9*interval_lengths] = 0
		is_mat_match[sibpair_index, mat_match_data[:, 1]>=0.9*interval_lengths] = 1
		is_pat_match[sibpair_index, pat_match_data[:, 0]>=0.9*interval_lengths] = 0
		is_pat_match[sibpair_index, pat_match_data[:, 1]>=0.9*interval_lengths] = 1
				
	return is_mat_match, is_pat_match

def calculate_pvalues(sibpairs, is_mat_match, is_pat_match):

	is_aff_aff = np.array([sp.num_affected==2 for sp in sibpairs])
	is_aff_typ = np.array([sp.num_affected==1 for sp in sibpairs])
	is_typ_typ = np.array([sp.num_affected==0 for sp in sibpairs])
	print(np.sum(is_aff_aff), np.sum(is_aff_typ), np.sum(is_typ_typ))

	# pos, UU/AU/AA/UUvsAU/AAvsAU/UU+AAvsAU
	pvalues = np.ones((is_mat_match.shape[1], 6))
	contingency = np.zeros((is_mat_match.shape[1], 3, 2))
	for interval_index in range(is_mat_match.shape[1]):
		# UU vs 0.5
		try:
			#print(np.sum(is_mat_match[is_typ_typ, interval_index]==1), np.sum(is_pat_match[is_typ_typ, interval_index]==1), 
			#	np.sum(is_mat_match[is_typ_typ, interval_index]!=-1), np.sum(is_pat_match[is_typ_typ, interval_index]!=-1))
			pvalues[interval_index, 0] = scipy.stats.binom_test(
				np.sum(is_mat_match[is_typ_typ, interval_index]==1) + np.sum(is_pat_match[is_typ_typ, interval_index]==1), 
				np.sum(is_mat_match[is_typ_typ, interval_index]!=-1) + np.sum(is_pat_match[is_typ_typ, interval_index]!=-1), 
				p=0.5, alternative='greater')
			contingency[interval_index, 0, 0] = np.sum(is_mat_match[is_typ_typ, interval_index]==1) + np.sum(is_pat_match[is_typ_typ, interval_index]==1)
			contingency[interval_index, 0, 1] = np.sum(is_mat_match[is_typ_typ, interval_index]!=-1) + np.sum(is_pat_match[is_typ_typ, interval_index]!=-1)
		except:
			pass

		# UA vs 0.5
		try:
			pvalues[interval_index, 1] = scipy.stats.binom_test(
				np.sum(is_mat_match[is_aff_typ, interval_index]==1) + np.sum(is_pat_match[is_aff_typ, interval_index]==1), 
				np.sum(is_mat_match[is_aff_typ, interval_index]!=-1) + np.sum(is_pat_match[is_aff_typ, interval_index]!=-1), 
				p=0.5, alternative='less')
			contingency[interval_index, 1, 0] = np.sum(is_mat_match[is_aff_typ, interval_index]==1) + np.sum(is_pat_match[is_aff_typ, interval_index]==1)
			contingency[interval_index, 1, 1] = np.sum(is_mat_match[is_aff_typ, interval_index]!=-1) + np.sum(is_pat_match[is_aff_typ, interval_index]!=-1)
		except:
			pass

		# AA vs 0.5
		try:
			pvalues[interval_index, 2] = scipy.stats.binom_test(
				np.sum(is_mat_match[is_aff_aff, interval_index]==1) + np.sum(is_pat_match[is_aff_aff, interval_index]==1), 
				np.sum(is_mat_match[is_aff_aff, interval_index]!=-1) + np.sum(is_pat_match[is_aff_aff, interval_index]!=-1), 
				p=0.5, alternative='greater')
			contingency[interval_index, 2, 0] = np.sum(is_mat_match[is_aff_aff, interval_index]==1) + np.sum(is_pat_match[is_aff_aff, interval_index]==1)
			contingency[interval_index, 2, 1] = np.sum(is_mat_match[is_aff_aff, interval_index]!=-1) + np.sum(is_pat_match[is_aff_aff, interval_index]!=-1)
		except:
			pass

		# UU vs AU
		try:
			pvalues[interval_index, 3] = scipy.stats.chi2_contingency(
				[[np.sum(is_mat_match[is_typ_typ, interval_index]==1) + np.sum(is_pat_match[is_typ_typ, interval_index]==1),
				  np.sum(is_mat_match[is_typ_typ, interval_index]==0) + np.sum(is_pat_match[is_typ_typ, interval_index]==0)],
				 [np.sum(is_mat_match[is_aff_typ, interval_index]==1) + np.sum(is_pat_match[is_aff_typ, interval_index]==1),
				  np.sum(is_mat_match[is_aff_typ, interval_index]==0) + np.sum(is_pat_match[is_aff_typ, interval_index]==0)]])[1]
		except:
			pass

		# AA vs AU
		try:
			pvalues[interval_index, 4] = scipy.stats.chi2_contingency(
				[[np.sum(is_mat_match[is_aff_aff, interval_index]==1) + np.sum(is_pat_match[is_aff_aff, interval_index]==1),
				  np.sum(is_mat_match[is_aff_aff, interval_index]==0) + np.sum(is_pat_match[is_aff_aff, interval_index]==0)],
				 [np.sum(is_mat_match[is_aff_typ, interval_index]==1) + np.sum(is_pat_match[is_aff_typ, interval_index]==1),
				  np.sum(is_mat_match[is_aff_typ, interval_index]==0) + np.sum(is_pat_match[is_aff_typ, interval_index]==0)]])[1]
		except:
			pass

		# AA+UU vs AU
		try:
			pvalues[interval_index, 5] = scipy.stats.chi2_contingency(
				[[np.sum(is_mat_match[is_aff_aff, interval_index]==1) + np.sum(is_pat_match[is_aff_aff, interval_index]==1) + \
				  np.sum(is_mat_match[is_typ_typ, interval_index]==1) + np.sum(is_pat_match[is_typ_typ, interval_index]==1),
				  np.sum(is_mat_match[is_aff_aff, interval_index]==0) + np.sum(is_pat_match[is_aff_aff, interval_index]==0) + \
				  np.sum(is_mat_match[is_typ_typ, interval_index]==0) + np.sum(is_pat_match[is_typ_typ, interval_index]==0)],
				 [np.sum(is_mat_match[is_aff_typ, interval_index]==1) + np.sum(is_pat_match[is_aff_typ, interval_index]==1),
				  np.sum(is_mat_match[is_aff_typ, interval_index]==0) + np.sum(is_pat_match[is_aff_typ, interval_index]==0)]])[1]
		except:
			pass

	return pvalues, contingency


if __name__ == "__main__":

	# pull sample info
	sample_to_affected, sample_to_sex = dict(), dict()
	for ped_file in ped_files:
		aff, sex = pull_phenotype_ped(ped_file)
		sample_to_affected.update(aff)
		sample_to_sex.update(sex)

	# pull sibpairs
	family_to_inds, sibpairs = dict(), []
	for phase_dir, identicals_file in zip(phase_dirs, identicals_files):
		print(phase_dir)
		faminds, sps = pull_sibpairs(phase_dir, identicals_file, sample_to_affected, sample_to_sex)
		duplicate_families = set()
		for famkey, inds in faminds.items():
			if famkey not in family_to_inds:
				family_to_inds[famkey] = inds
			else:
				duplicate_families.add(famkey)

		sibpairs.extend([x for x in sps if x.family not in duplicate_families])
		print('duplicates', len(duplicate_families))

	print('Overall')
	print('families', len(family_to_inds))
	print('sibpairs', len(sibpairs))
	print('num_affected', Counter([x.num_affected for x in sibpairs]))

	with open('data/chrom_lengths%s.json' % build, 'r') as f:
		chrom_lengths = json.load(f)
	
	for chrom in chroms:
		outfile = '%s/chr.%s.IST.pvalues.be.intervals.%d' % (out_dir, chrom, interval)

		interval_bins = np.arange(0, chrom_lengths[chrom]+interval, interval)
		print('intervals', len(interval_bins)-1)

		is_mat_match, is_pat_match = pull_sibpair_matches(phase_dirs, chrom, sibpairs, family_to_inds, interval_bins)

		pvalues, contingency = calculate_pvalues(sibpairs, is_mat_match, is_pat_match)
		print('pvalues computed')

		print(pvalues.shape)
		np.save(outfile, pvalues)
		np.save(outfile + '.regions', interval_bins)

		if save_contingency:
			np.save(outfile + '.contingency', contingency)
		print('results saved to %s' % outfile)


