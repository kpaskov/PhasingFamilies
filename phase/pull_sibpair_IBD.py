from collections import defaultdict, Counter, namedtuple
import numpy as np
import math
from itertools import combinations, product
import scipy.stats as stats
from os import listdir
import argparse
import json
from input_output import parse_phase_file
from numpyencoder import NumpyEncoder


parser = argparse.ArgumentParser(description='Pulls sibpair similarity from phasing output.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data for the cohort in .npy format.')
parser.add_argument('--include_X', action='store_true', default=False, help='Include chrX when calculating chromosome-level IBD.')

chroms = [str(x) for x in range(1, 23)]

args = parser.parse_args()

if args.include_X:
	chroms += ['X']

# pull sibpairs from phase files
phase_files = [x for x in listdir('%s/phase/inheritance_patterns' % args.data_dir) if x.endswith('.phased.bed')]
families = []
sibpairs = []

for filename in phase_files:
	family = filename[:-11]

	phase_data = parse_phase_file('%s/phase/inheritance_patterns/%s.phased.bed' % (args.data_dir, family), chroms)
	inds, is_standard = next(phase_data)

	# we don't handle nonstandaard families in this script
	if is_standard:
		for child1, child2 in combinations(inds[2:], 2):
			families.append(family)
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

	for i, (family, sibpair) in enumerate(zip(families, sibpairs)):
		if i%100==0:
			print(i, end=' ')

		phase_data = parse_phase_file('%s/phase/inheritance_patterns/%s.phased.bed' % (args.data_dir, family), chroms)

		inds, is_standard = next(phase_data)
		child1_index = inds.index(sibpair['sibling1'])
		child2_index = inds.index(sibpair['sibling2'])
		sibpair_chroms = set()

		for segment in phase_data:
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

	print('Discarding %d sibpairs missing phased chroms' % np.sum(~has_all_chroms))
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

print('Calculating autosomal IBD')
has_all_chroms, mat_match, mat_mismatch, mat_unknown, \
pat_match, pat_mismatch, pat_unknown, \
both_match, both_mismatch, both_unknown = load_phase_data(set([str(x) for x in range(1, 23)]), sibpairs)

for i, sibpair in enumerate(sibpairs):
	if has_all_chroms[i]:
		sibpair['maternal_ibd'] = mat_match[i]/(mat_match[i]+mat_mismatch[i])
		sibpair['maternal_unknown_fraction'] = mat_unknown[i]/(mat_match[i]+mat_mismatch[i]+mat_unknown[i])

		sibpair['paternal_ibd'] = pat_match[i]/(pat_match[i]+pat_mismatch[i])
		sibpair['paternal_unknown_fraction'] = pat_unknown[i]/(pat_match[i]+pat_mismatch[i]+pat_unknown[i])

		sibpair['matxpat_ibd'] = both_match[i]/(both_match[i]+both_mismatch[i])
		sibpair['matxpat_unknown_fraction'] = both_unknown[i]/(both_match[i]+both_mismatch[i]+both_unknown[i])

	sibpair['maternal_ibd_chroms'] = [None]*23
	sibpair['maternal_unknown_fraction_chroms'] = [None]*23
	sibpair['paternal_ibd_chroms'] = [None]*23
	sibpair['paternal_unknown_fraction_chroms'] = [None]*23
	sibpair['matxpat_ibd_chroms'] = [None]*23
	sibpair['matxpat_unknown_fraction_chroms'] = [None]*23

print('Calculating chromosome-level IBD')

for j, chrom in enumerate([str(x) for x in range(1, 23)] + ['X']):
	print('chrom', chrom)
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

for sibpair in sibpairs:
	mat_ibd, pat_ibd = sibpair.get('maternal_ibd', None), sibpair.get('paternal_ibd', None)
	sibpair['is_identical'] = None if (mat_ibd is None or pat_ibd is None) else (mat_ibd > 0.8) and (pat_ibd > 0.8)
	sibpair['is_full_sibling'] = None if (mat_ibd is None or pat_ibd is None) else (mat_ibd > 0.2) and (pat_ibd > 0.2)

with open('%s/phase/sibpairs.json' % args.data_dir, 'w+') as f:
	json.dump(sibpairs, f, indent=4, cls=NumpyEncoder)

