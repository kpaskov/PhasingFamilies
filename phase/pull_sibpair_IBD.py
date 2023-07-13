from collections import defaultdict, Counter, namedtuple
import numpy as np
import math
from itertools import combinations, product
import scipy.stats as stats
from os import listdir
import argparse
import json
from input_output import PhaseData
from numpyencoder import NumpyEncoder
import sys
from qc import OutlierDetector

parser = argparse.ArgumentParser(description='Pulls sibpair similarity from phasing output.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data for the cohort in .npy format.')
parser.add_argument('--phase_name', type=str, default=None, help='Name for the phase attempt.')
args = parser.parse_args()

phase_data = PhaseData(args.data_dir, args.phase_name)

# pull sibpairs from phase files
families = []
sibpairs = []
indices = []

for family in phase_data.get_phased_families():
	inds = phase_data.get_phase_info(family)['individuals']
	is_standard = phase_data.is_standard_family(family)

	# we don't handle nonstandard families in this script
	if is_standard:
		for child1, child2 in combinations(inds[2:], 2):
			families.append(family)
			indices.append((inds.index(min(child1, child2)), inds.index(max(child1, child2))))
			sibpairs.append({
							'family': family,
							'sibling1': min(child1, child2),
							'sibling2': max(child1, child2)
							})
                    
print('sibpairs found', len(sibpairs))


# -------------------------- Load phase data --------------------------
def load_phase_data(chroms, sibpairs):
	mat_match = np.zeros((len(sibpairs),), dtype=int)
	mat_mismatch = np.zeros((len(sibpairs),), dtype=int)
	mat_unknown = np.zeros((len(sibpairs),), dtype=int)

	pat_match = np.zeros((len(sibpairs),), dtype=int)
	pat_mismatch = np.zeros((len(sibpairs),), dtype=int)
	pat_unknown = np.zeros((len(sibpairs),), dtype=int)

	both_match = np.zeros((len(sibpairs),), dtype=int)
	both_mismatch = np.zeros((len(sibpairs),), dtype=int)
	both_unknown = np.zeros((len(sibpairs),), dtype=int)

	has_all_chroms = np.zeros((len(sibpairs),), dtype=bool)

	for i, (family, sibpair, (child1_index, child2_index)) in enumerate(zip(families, sibpairs, indices)):
		if i%100==0:
			print(i, end=' ')
			sys.stdout.flush()

		sibpair_chroms = set()
		for segment in phase_data.parse_phase_file(family, chroms):
			mat1, mat2 = segment.mat_phase[child1_index], segment.mat_phase[child2_index]
			pat1, pat2 = segment.pat_phase[child1_index], segment.pat_phase[child2_index]
			length = segment.length()
			sibpair_chroms.add(segment.chrom)

			# we don't handle UPD in this script
			is_mat_upd = segment.is_mat_upd(child1_index) or segment.is_mat_upd(child2_index)
			is_pat_upd = segment.is_pat_upd(child1_index) or segment.is_pat_upd(child2_index)
			assert (not is_mat_upd) and (not is_pat_upd)

			is_hts = segment.is_hts()

			#if mat1==-1 or mat2==-1 or is_hts:
			if mat1==-1 or mat2==-1:
				mat_unknown[i] += length
			elif mat1==mat2:
				mat_match[i] += length
			else:
				mat_mismatch[i] += length

			#if pat1==-1 or pat2==-1 or is_hts:
			if pat1==-1 or pat2==-1:
				pat_unknown[i] += length
			elif pat1==pat2:
				pat_match[i] += length
			else:
				pat_mismatch[i] += length
                            
			#if mat1==-1 or mat2==-1 or pat1==-1 or pat2==-1 or is_hts:
			if mat1==-1 or mat2==-1 or pat1==-1 or pat2==-1:
				both_unknown[i] += length
			elif mat1==mat2 and pat1==pat2:
				both_match[i] += length
			else:
				both_mismatch[i] += length           

		has_all_chroms[i] = len(chroms) == len(sibpair_chroms)

	print('\n%d sibpairs missing phased chroms %s' % (np.sum(~has_all_chroms), str(chroms)))
	if np.sum(~has_all_chroms) != len(has_all_chroms):
		print([sibpairs[i]['family'] for i in np.where(~has_all_chroms)[0]])
    
    #mat_scores = mat_match/(mat_match+mat_mismatch)
    #pat_scores = pat_match/(pat_match+pat_mismatch)
    #both_scores = both_match/(both_match+both_mismatch)
    #mat_unknown_fraction = mat_unknown/(mat_match+mat_mismatch+mat_unknown)
    #pat_unknown_fraction = pat_unknown/(pat_match+pat_mismatch+pat_unknown)
    
    #mat_scores[~has_all_chroms] = -1
    #pat_scores[~has_all_chroms] = -1
    #both_scores[~has_all_chroms] = -1

	return has_all_chroms, mat_match, mat_mismatch, mat_unknown, \
		pat_match, pat_mismatch, pat_unknown, \
		both_match, both_mismatch, both_unknown

print('Calculating dataset IBD')
has_all_chroms, mat_match, mat_mismatch, mat_unknown, \
pat_match, pat_mismatch, pat_unknown, \
both_match, both_mismatch, both_unknown = load_phase_data(phase_data.chroms, sibpairs)

for i, sibpair in enumerate(sibpairs):
	sibpair['is_fully_phased'] = has_all_chroms[i]

	if sibpair['is_fully_phased']:
		sibpair['maternal_ibd'] = mat_match[i]/(mat_match[i]+mat_mismatch[i])
		sibpair['maternal_unknown_fraction'] = mat_unknown[i]/(mat_match[i]+mat_mismatch[i]+mat_unknown[i])

		sibpair['paternal_ibd'] = pat_match[i]/(pat_match[i]+pat_mismatch[i])
		sibpair['paternal_unknown_fraction'] = pat_unknown[i]/(pat_match[i]+pat_mismatch[i]+pat_unknown[i])

		sibpair['matxpat_ibd'] = both_match[i]/(both_match[i]+both_mismatch[i])
		sibpair['matxpat_unknown_fraction'] = both_unknown[i]/(both_match[i]+both_mismatch[i]+both_unknown[i])

		sibpair['is_identical'] = (sibpair['maternal_ibd'] > 0.8) and (sibpair['paternal_ibd'] > 0.8)
	else:
		sibpair['maternal_ibd'] = None
		sibpair['maternal_unknown_fraction'] = None
		sibpair['paternal_ibd'] = None
		sibpair['paternal_unknown_fraction'] = None
		sibpair['matxpat_ibd'] = None
		sibpair['matxpat_unknown_fraction'] = None

		sibpair['is_identical'] = None
		
	sibpair['maternal_ibd_chroms'] = [None]*23
	sibpair['maternal_unknown_fraction_chroms'] = [None]*23
	sibpair['paternal_ibd_chroms'] = [None]*23
	sibpair['paternal_unknown_fraction_chroms'] = [None]*23
	sibpair['matxpat_ibd_chroms'] = [None]*23
	sibpair['matxpat_unknown_fraction_chroms'] = [None]*23
		
	sibpair['is_ibd_outlier'] = None

print('Identifying IBD outliers')
is_identical = np.array([x['is_identical'] if x['is_identical'] is not None else False for x in sibpairs])
mat_ibd = np.array([x['maternal_ibd'] for x in sibpairs])
pat_ibd = np.array([x['paternal_ibd'] for x in sibpairs])

if len(phase_data.chroms)==22:

	detector = OutlierDetector(mat_ibd[has_all_chroms & ~is_identical], pat_ibd[has_all_chroms & ~is_identical],
		0.1)
	is_outlier = detector.predict_outliers(mat_ibd[has_all_chroms], pat_ibd[has_all_chroms])
else:
	is_outlier = np.zeros((len(sibpairs),))
for i, sibpair in enumerate([x for x, hi in zip(sibpairs, has_all_chroms) if hi]):
	sibpair['is_ibd_outlier'] = is_outlier[i]
print('outliers marked', np.sum(is_outlier))

print('Calculating chromosome-level IBD')
for j, chrom in enumerate([str(x) for x in range(1, 23)] + ['X']):
	print('chrom', chrom)
	if chrom in phase_data.chroms:
		has_all_chroms, mat_match, mat_mismatch, mat_unknown, \
		pat_match, pat_mismatch, pat_unknown, \
		both_match, both_mismatch, both_unknown = load_phase_data(set([chrom]), sibpairs)

		for i, sibpair in enumerate(sibpairs):
			if has_all_chroms[i]:
				sibpair['maternal_ibd_chroms'][j] = mat_match[i]/(mat_match[i]+mat_mismatch[i])
				sibpair['maternal_unknown_fraction_chroms'][j] = mat_unknown[i]/(mat_match[i]+mat_mismatch[i]+mat_unknown[i])

				sibpair['paternal_ibd_chroms'][j] = pat_match[i]/(pat_match[i]+pat_mismatch[i])
				sibpair['paternal_unknown_fraction_chroms'][j] = pat_unknown[i]/(pat_match[i]+pat_mismatch[i]+pat_unknown[i])

				sibpair['matxpat_ibd_chroms'][j] = both_match[i]/(both_match[i]+both_mismatch[i])
				sibpair['matxpat_unknown_fraction_chroms'][j] = both_unknown[i]/(both_match[i]+both_mismatch[i]+both_unknown[i])

with open('%s/sibpairs.json' % phase_data.phase_dir, 'w+') as f:
	json.dump(sibpairs, f, indent=4, cls=NumpyEncoder)

