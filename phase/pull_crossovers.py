import numpy as np
from collections import namedtuple, defaultdict, Counter
from itertools import product, combinations
import json
import argparse
import input_output
import os
import traceback
from numpyencoder import NumpyEncoder
import traceback
from input_output import PhaseData
from qc import OutlierDetector

parser = argparse.ArgumentParser(description='Pull crossovers from phasing output.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data for the cohort in .npy format.')
parser.add_argument('--phase_name', type=str, default=None, help='Name for the phase attempt.')

args = parser.parse_args()

phase_data = PhaseData(args.data_dir, args.phase_name)

# extracts all recombination points from phase
Recombination = namedtuple('Recombination', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_hts'])


def pull_phase_data_into_arrays(family):
	# pull phase data
	mat_phases, pat_phases = [], []
	loss_regions = []
	is_mat_upds, is_pat_upds = [], []
	chroms, starts, ends = [], [], []
	is_htss = []
	for segment in phase_data.parse_phase_file(family):
		chroms.append(segment.chrom)
		mat_phases.append(segment.mat_phase)
		pat_phases.append(segment.pat_phase)
		starts.append(segment.start_pos)
		ends.append(segment.end_pos)
		loss_regions.append(segment.loss_region)
		is_mat_upds.append(np.any(segment.is_mat_upd()))
		is_pat_upds.append(np.any(segment.is_pat_upd()))
		is_htss.append(segment.is_hts())

	mat_phases = np.array(mat_phases).T
	pat_phases = np.array(pat_phases).T
	loss_regions = np.array(loss_regions)
	is_mat_upds = np.array(np.any(is_mat_upds))
	is_pat_upds = np.array(np.any(is_pat_upds))
	starts = np.array(starts)
	ends = np.array(ends)
	is_htss = np.array(is_htss)

	# if this is a hard to sequences region, we don't know the exact location of crossovers
	mat_phases[:, is_htss] = -1
	pat_phases[:, is_htss] = -1

	# if there's UPD, it's not a crossover
	mat_phases[:, is_mat_upds] = -1
	pat_phases[:, is_pat_upds] = -1

	return chroms, starts, ends, mat_phases, pat_phases, is_htss

def pull_recombinations(family_id, phases, is_htss, chroms, starts, ends, individuals, is_mat):
	recombinations = []
	for chrom in phase_data.chroms:
		# block out 
		is_chrom = np.array([x==chrom for x in chroms])
		for index, individual in enumerate(individuals):
			change_indices = np.where(is_chrom[:-1] & is_chrom[1:] & (phases[index, :-1] != phases[index, 1:]))[0]+1

			if change_indices.shape[0]>0:
				current_index = change_indices[0]
				for next_index in change_indices[1:]:
					if phases[index, current_index-1] != -1 and phases[index, current_index] != -1:
						# we know exactly where the recombination happened
						recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[current_index]), 
							(individual, individuals[2]), 
							is_mat, not is_mat,
							is_htss[current_index]
							))

					elif phases[index, current_index-1] != -1 and phases[index, current_index] == -1 and phases[index, current_index-1] != phases[index, next_index]:
						# there's a region where the recombination must have occured
						recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[next_index]), 
							(individual, individuals[2]), 
							is_mat, not is_mat,
							np.any(is_htss[current_index:next_index])))
					current_index = next_index

				if phases[index, current_index-1] != -1 and phases[index, current_index] != -1:
					# we know exactly where the recombination happened
					recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[current_index]), 
						(individual, individuals[2]), 
						is_mat, not is_mat,
						is_htss[current_index]))
	return recombinations

Crossover = namedtuple('Crossover', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'is_hts', 'recombinations'])
GeneConversion = namedtuple('GeneConversion', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'is_hts', 'recombinations'])

def match_recombinations(recombinations, chrom, family_id, child, is_mat):
    cutoff = 161332 if is_mat else 154265 # pulled from spark 0.01 quantile

    gene_conversions = []
    crossovers = []
    
    rs = [r for r in recombinations if r.chrom==chrom and r.child==child and r.is_mat==is_mat]
    dists = np.array([r2.start_pos-r1.end_pos for r1, r2 in zip(rs[:-1], rs[1:])])
   
    breaks = [0] + (np.where(dists>cutoff)[0]+1).tolist() + [len(rs)]
    for break_start, break_end in zip(breaks[:-1], breaks[1:]):
        r_group = rs[break_start:break_end]
        if len(r_group)>0:
            if len(r_group)%2 == 0:
                gene_conversions.append(GeneConversion(family_id, chrom, 
                                                       r_group[0].start_pos, r_group[-1].end_pos, child,
                                                       is_mat, not is_mat, 
                                                       len(r_group)>2, # is_complex
                                                       np.any([r.is_hts for r in r_group]), # is_hts
                                                       [r._asdict() for r in r_group]))
            else:
                crossovers.append(Crossover(family_id, chrom,
                                           r_group[0].start_pos, r_group[-1].end_pos, child,
                                           is_mat, not is_mat, 
                                           len(r_group)>1, # is_complex
                                           np.any([r.is_hts for r in r_group]), # is_hts
                                           [r._asdict() for r in r_group]))
                    
    return gene_conversions, crossovers

all_recombinations = []
all_crossovers = []
all_gene_conversions = []

sibpairs = phase_data.get_sibpairs()
for sibpair in sibpairs:
	family = sibpair['family']
	print(family)

	individuals = phase_data.get_phase_info(family)['individuals']
	if phase_data.is_standard_family(family) and sibpair['is_fully_phased']:
		chroms, starts, ends, mat_phases, pat_phases, is_htss = pull_phase_data_into_arrays(family)

		# pull recombinations
		mat_recombinations = pull_recombinations(family, mat_phases, is_htss, chroms, starts, ends, individuals, True)
		pat_recombinations = pull_recombinations(family, pat_phases, is_htss, chroms, starts, ends, individuals, False)
		recombinations = mat_recombinations + pat_recombinations

		# identify crossovers and gene conversions
		gene_conversions, crossovers = [], []
		children = set([x.child for x in recombinations])
		for chrom in phase_data.chroms:
			for child in children:
				gc, co = match_recombinations(recombinations, chrom, family, child, True)
				gene_conversions.extend(gc)
				crossovers.extend(co)

				gc, co = match_recombinations(recombinations, chrom, family, child, False)
				gene_conversions.extend(gc)
				crossovers.extend(co)

		print('gc', len(gene_conversions), 'co', len(crossovers))

		num_children = len(individuals)-2
		gc = len(gene_conversions)/num_children
		cr = len(crossovers)/num_children
		print('avg gene conversion', gc, 'avg crossover', cr)

		all_recombinations.extend(recombinations)
		all_crossovers.extend(crossovers)
		all_gene_conversions.extend(gene_conversions)


#with open('%s/recombinations.json' % phase_data.phase_dir, 'w+') as f:
#	json.dump([c._asdict() for c in all_recombinations], f, indent=4, cls=NumpyEncoder)	

with open('%s/crossovers.json' % phase_data.phase_dir, 'w+') as f:
	json.dump([c._asdict() for c in all_crossovers], f, indent=4, cls=NumpyEncoder)

with open('%s/gene_conversions.json' % phase_data.phase_dir, 'w+') as f:
	json.dump([gc._asdict() for gc in all_gene_conversions], f, indent=4, cls=NumpyEncoder)

# Now identify outliers
sibpair_to_num_mat_crossovers = defaultdict(int)
sibpair_to_num_pat_crossovers = defaultdict(int)

sibpair_to_index = dict([((x['family'], x['sibling1'], x['sibling2']), i) for i, x in enumerate(sibpairs)])
mat_crossovers, pat_crossovers = np.zeros((len(sibpairs),), dtype=int), np.zeros((len(sibpairs),), dtype=int)
for co in all_crossovers:
	key = (co.family, co.child[0], co.child[1])
	if key not in sibpair_to_index:
		key = (co.family, co.child[1], co.child[0])
	if co.is_mat:
		mat_crossovers[sibpair_to_index[key]] += 1
	if co.is_pat:
		pat_crossovers[sibpair_to_index[key]] += 1

is_fully_phased = np.array([x['is_fully_phased'] for x in sibpairs], dtype=bool)
is_ibd_outlier = np.array([x['is_ibd_outlier'] if x['is_ibd_outlier'] is not None else False for x in sibpairs], dtype=bool)

if len(phase_data.chroms)==22 and len(sibpairs)>5:
	is_way_out = (mat_crossovers > 3*np.median(mat_crossovers)) | (pat_crossovers > 3*np.median(pat_crossovers))
	detector = OutlierDetector(mat_crossovers[is_fully_phased & ~is_ibd_outlier & ~is_way_out], pat_crossovers[is_fully_phased & ~is_ibd_outlier & ~is_way_out], 
		10 if np.median(mat_crossovers)>10 else 1)
	is_outlier = detector.predict_outliers(mat_crossovers, pat_crossovers)
else:
	is_outlier = np.zeros((len(sibpairs),), dtype=bool)

for i, sibpair in enumerate(sibpairs):
	if sibpair['is_fully_phased']:
		sibpair['maternal_crossovers'] = mat_crossovers[i]
		sibpair['paternal_crossovers'] = pat_crossovers[i]
		sibpair['is_crossover_outlier'] = is_outlier[i]
	else:
		sibpair['maternal_crossovers'] = None
		sibpair['paternal_crossovers'] = None
		sibpair['is_crossover_outlier'] = None

print('outliers marked', np.sum(is_outlier))

with open('%s/sibpairs.json' % phase_data.phase_dir, 'w+') as f:
	json.dump(sibpairs, f, indent=4, cls=NumpyEncoder)


