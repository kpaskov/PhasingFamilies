from collections import defaultdict, Counter
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations, product
import scipy.stats as stats


ihart_family_sizes = [3, 4, 5, 6]
ihart_phase_dir = 'phased_ihart'

ssc_family_sizes = [3, 4]
ssc_phase_dir = 'phased_ssc'

spark_family_sizes = [4, 5, 6, 7]
spark_phase_dir = 'phased_spark'

# need affected/unaffected information
# these files can be found on sherlock
# '../data/v34.vcf.ped' = /scratch/PI/dpwall/DATA/iHART/vcf/v3.4/v34.vcf.ped
# '../data/ssc.ped' = /scratch/PI/dpwall/DATA/iHART/vcf/SSC/ssc.ped

ped_files = ['data/v34.vcf.ped', 'data/ssc.ped', 'data/spark.ped']
# Affection (0=unknown; 1=unaffected; 2=affected)
child_id_to_sex = dict()

for ped_file in ped_files:
	with open(ped_file, 'r') as f:
		for line in f:
			pieces = line.strip().split('\t')
			if len(pieces) >= 6:
				fam_id, child_id, f_id, m_id, sex, disease_status = pieces[0:6]
				child_id_to_sex[child_id] = sex


# -------------------------- Load phase data --------------------------
def load_phase_data(chroms, phase_dir, family_sizes, ignore_mask=False):
	sibpair_to_mat_match = defaultdict(int)
	sibpair_to_mat_mismatch = defaultdict(int)
	sibpair_to_pat_match = defaultdict(int)
	sibpair_to_pat_mismatch = defaultdict(int)

	for chrom in chroms:
		print(chrom, end=' ')

		# pull families
		family_to_individuals = dict()
		for j in family_sizes:
			with open('%s/chr.%s.familysize.%d.families.txt' % (phase_dir, chrom, j), 'r')  as f: 
				next(f) # skip header
				for line in f:
					pieces = line.strip().split('\t')
					family_key = pieces[0]
					family_to_individuals[family_key] = pieces[1:(1+j)]


		# now read phase info
		for j in family_sizes:
			# load deletions
			with open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, j), 'r')  as f:
				next(f) # skip header

				for line in f:
					pieces = line.strip().split('\t')
					family_key = pieces[0]
					inheritance_state = [int(x) for x in pieces[1:(2+(j*2))]]
					start_pos, end_pos = [int(x) for x in pieces[(2+(j*2)):(4+(j*2))]]
					length = end_pos - start_pos + 1

					has_m1, has_m2, has_p1, has_p2 = [], [], [], []
					children = family_to_individuals[family_key][2:]
					if inheritance_state[-1] == 0 or ignore_mask:
						for i, child in enumerate(children):
							if inheritance_state[4+(2*i)] == 0:
								has_m1.append(child)
							elif inheritance_state[4+(2*i)] == 1:
								has_m2.append(child)
							if inheritance_state[5+(2*i)] == 0:
								has_p1.append(child)
							elif inheritance_state[5+(2*i)] == 1:
								has_p2.append(child)

					# matches
					for c1, c2 in combinations(has_m1, 2):
						sibpair_to_mat_match[(min(c1, c2), max(c1, c2))] += length
					for c1, c2 in combinations(has_m2, 2):
						sibpair_to_mat_match[(min(c1, c2), max(c1, c2))] += length
					for c1, c2 in combinations(has_p1, 2):
						sibpair_to_pat_match[(min(c1, c2), max(c1, c2))] += length
					for c1, c2 in combinations(has_p2, 2):
						sibpair_to_pat_match[(min(c1, c2), max(c1, c2))] += length

					# mismatches
					for c1, c2 in product(has_m1, has_m2):
						sibpair_to_mat_mismatch[(min(c1, c2), max(c1, c2))] += length
					for c1, c2 in product(has_p1, has_p2):
						sibpair_to_pat_mismatch[(min(c1, c2), max(c1, c2))] += length
	return sibpair_to_mat_match, sibpair_to_mat_mismatch, sibpair_to_pat_match, sibpair_to_pat_mismatch

def find_outliers(mat_match, mat_mismatch, pat_match, pat_mismatch, is_X=False):
	# very naive method - could use some work
	if is_X:
		m = max(np.max(mat_match+mat_mismatch), np.max(pat_match+pat_mismatch))
		pat_ok = (pat_match < 0.1*m) | (pat_mismatch < 0.1*m)
		return ~pat_ok | (mat_match+mat_mismatch < 0.5*m) 
	else:
		m = max(np.max(mat_match+mat_mismatch), np.max(pat_match+pat_mismatch))
		return (mat_match+mat_mismatch < 0.5*m) | (pat_match+pat_mismatch < 0.5*m) 

def calculate_sibpair_similarity(chroms, phase_dir, family_sizes):
	print(phase_dir)
	sibpair_to_mat_match, sibpair_to_mat_mismatch, sibpair_to_pat_match, sibpair_to_pat_mismatch = load_phase_data(chroms, phase_dir, family_sizes, ignore_mask=True)

	sibpairs = sorted(set(sibpair_to_mat_match.keys()))
	mat_match = np.array([sibpair_to_mat_match[k] for k in sibpairs])
	mat_mismatch = np.array([sibpair_to_mat_mismatch[k] for k in sibpairs])
	pat_match = np.array([sibpair_to_pat_match[k] for k in sibpairs])
	pat_mismatch = np.array([sibpair_to_pat_mismatch[k] for k in sibpairs])

	# remove outliers
	is_outlier = find_outliers(mat_match, mat_mismatch, pat_match, pat_mismatch, is_X=chroms==['X'])
	print('Removing %d outliers' % np.sum(is_outlier))

	sibpairs = [sibpairs[i] for i in np.where(~is_outlier)[0]]
	mat_match = mat_match[~is_outlier]
	mat_mismatch = mat_mismatch[~is_outlier]
	pat_match = pat_match[~is_outlier]
	pat_mismatch = pat_mismatch[~is_outlier]

	# calculate similarity score
	mat_scores = mat_match/(mat_match+mat_mismatch)
	pat_scores = pat_match/(pat_match+pat_mismatch)

	return sibpairs, mat_scores, pat_scores, mat_match, mat_mismatch, pat_match, pat_mismatch

# Individual chroms
sibpair_to_chrom_count = defaultdict(int)
for chrom in [str(x) for x in range(1, 23)] + ['X']:
	with open('sibpair_similarity/sibpair_similarity_scores%s_full_genome.txt' % chrom, 'w+') as f:
		f.write('Sibling1\tSibling2\tDataset\tMaternal-Similarity\tPaternal-Similarity\tMat-Bases-Match\tMat-Bases-Mismatch\tPat-Bases-Match\tPat-Bases-Mismatch\n')

		# iHART
		print('iHART')
		sibpairs, mat_scores, pat_scores, mat_match, mat_mismatch, pat_match, pat_mismatch = calculate_sibpair_similarity([chrom], ihart_phase_dir, ihart_family_sizes)
		for i, sibpair in enumerate(sibpairs):
			f.write('%s\t%s\t%s\t%f\t%f\t%d\t%d\t%d\t%d\n' % (sibpair[0], sibpair[1], 'iHART', mat_scores[i], pat_scores[i], mat_match[i], mat_mismatch[i], pat_match[i], pat_mismatch[i]))
			sibpair_to_chrom_count[sibpair] += 1

		# SSC
		print('SSC')
		sibpairs, mat_scores, pat_scores, mat_match, mat_mismatch, pat_match, pat_mismatch = calculate_sibpair_similarity([chrom], ssc_phase_dir, ssc_family_sizes)
		for i, sibpair in enumerate(sibpairs):
			f.write('%s\t%s\t%s\t%f\t%f\t%d\t%d\t%d\t%d\n' % (sibpair[0], sibpair[1], 'SSC', mat_scores[i], pat_scores[i], mat_match[i], mat_mismatch[i], pat_match[i], pat_mismatch[i]))
			sibpair_to_chrom_count[sibpair] += 1

		# SPARK
		print('SPARK')
		sibpairs, mat_scores, pat_scores, mat_match, mat_mismatch, pat_match, pat_mismatch = calculate_sibpair_similarity([chrom], spark_phase_dir, spark_family_sizes)
		for i, sibpair in enumerate(sibpairs):
			f.write('%s\t%s\t%s\t%f\t%f\t%d\t%d\t%d\t%d\n' % (sibpair[0], sibpair[1], 'Spark', mat_scores[i], pat_scores[i], mat_match[i], mat_mismatch[i], pat_match[i], pat_mismatch[i]))
			sibpair_to_chrom_count[sibpair] += 1

# All chroms
chroms = [str(x) for x in range(1, 23)] 
with open('sibpair_similarity/sibpair_similarity_scores_full_genome.txt', 'w+') as f:
	f.write('Sibling1\tSibling2\tDataset\tMaternal-Similarity\tPaternal-Similarity\tMat-Bases-Match\tMat-Bases-Mismatch\tPat-Bases-Match\tPat-Bases-Mismatch\n')

	# iHART
	print('iHART')
	sibpairs, mat_scores, pat_scores, mat_match, mat_mismatch, pat_match, pat_mismatch = calculate_sibpair_similarity(chroms, ihart_phase_dir, ihart_family_sizes)
	for i, sibpair in enumerate(sibpairs):
		if sibpair_to_chrom_count[sibpair] == 23:
			f.write('%s\t%s\t%s\t%f\t%f\t%d\t%d\t%d\t%d\n' % (sibpair[0], sibpair[1], 'iHART', mat_scores[i], pat_scores[i], mat_match[i], mat_mismatch[i], pat_match[i], pat_mismatch[i]))

	# SSC
	print('SSC')
	sibpairs, mat_scores, pat_scores, mat_match, mat_mismatch, pat_match, pat_mismatch = calculate_sibpair_similarity(chroms, ssc_phase_dir, ssc_family_sizes)
	for i, sibpair in enumerate(sibpairs):
		if sibpair_to_chrom_count[sibpair] == 23:
			f.write('%s\t%s\t%s\t%f\t%f\t%d\t%d\t%d\t%d\n' % (sibpair[0], sibpair[1], 'SSC', mat_scores[i], pat_scores[i], mat_match[i], mat_mismatch[i], pat_match[i], pat_mismatch[i]))

	# SPARK
	print('SPARK')
	sibpairs, mat_scores, pat_scores, mat_match, mat_mismatch, pat_match, pat_mismatch = calculate_sibpair_similarity(chroms, spark_phase_dir, spark_family_sizes)
	for i, sibpair in enumerate(sibpairs):
		if sibpair_to_chrom_count[sibpair] == 23:
			f.write('%s\t%s\t%s\t%f\t%f\t%d\t%d\t%d\t%d\n' % (sibpair[0], sibpair[1], 'Spark', mat_scores[i], pat_scores[i], mat_match[i], mat_mismatch[i], pat_match[i], pat_mismatch[i]))

