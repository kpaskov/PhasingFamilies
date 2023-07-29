from collections import defaultdict, namedtuple
import sys
import json
import numpy as np
import math
from os import listdir
from enum import Enum
from numpyencoder import NumpyEncoder
import argparse
from input_output import PhaseData

parser = argparse.ArgumentParser(description='Pull crossovers from phasing output.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data for the cohort in .npy format.')
parser.add_argument('--phase_name', type=str, default=None, help='Name for the phase attempt.')
parser.add_argument('--hts_loss_regions', type=str, nargs='+', default = [1], 
	help='These loss regions represent hard to sequence regions of the genome so we assume that we cant detect crossovers.')

args = parser.parse_args()

phase_data = PhaseData(args.data_dir, args.phase_name)

# read in deletions
Deletion = namedtuple('Deletion', ['family', 'chrom', 'start_pos', 'end_pos', 'length',
	'opt_start_pos', 'opt_end_pos', 'trans', 'notrans', 'is_mat', 'is_pat', 'is_hts', 'parent'])

def pull_phase_data_into_arrays(family):
	# pull phase data
	deletions = []
	mat_phases, pat_phases = [], []
	loss_regions = []
	chroms, starts, ends = [], [], []
	for segment in phase_data.parse_phase_file(family):
		chroms.append(segment.chrom)
		deletions.append(segment.deletions)
		mat_phases.append(segment.mat_phase)
		pat_phases.append(segment.pat_phase)
		starts.append(segment.start_pos)
		ends.append(segment.end_pos)
		loss_regions.append(segment.loss_region)

	deletions = np.array(deletions).T
	mat_phases = np.array(mat_phases).T
	pat_phases = np.array(pat_phases).T
	loss_regions = np.array(loss_regions)
	starts = np.array(starts)
	ends = np.array(ends)

	# if this is a hard to sequences region, we don't know the exact location of crossovers
	is_hts = np.zeros(loss_regions.shape, dtype=bool)
	for lr in args.hts_loss_regions:
		is_hts[loss_regions==lr] = True

	return chroms, starts, ends, deletions, mat_phases, pat_phases, is_hts

def pull_deletion_indices(data):
	assert (data[0] == 1) and (data[-1] == 1)
	change_indices = (np.where(data[:-1] != data[1:])[0]+1).tolist()
	group_start_indices = np.array([0] + change_indices)
	group_end_indices = np.array(change_indices + [data.shape[0]])
	group_states = data[group_start_indices]
	assert (group_states[0]==1) and (group_states[-1]==1)

	deletion_indices = []
	for group_index in np.where(group_states==0)[0]:
		start_index, end_index = group_start_indices[group_index], group_end_indices[group_index]
		#print(data[start_index:end_index])
		assert np.all(data[start_index:end_index]==0)

		if group_states[group_index-1] == 1:
			opt_start_index = start_index
		else:
			opt_start_index = group_start_indices[group_index-1]

		if group_states[group_index+1] == 1:
			opt_end_index = end_index
		else:
			opt_end_index = group_end_indices[group_index+1]

		deletion_indices.append((opt_start_index, start_index, end_index, opt_end_index))
	return deletion_indices

all_deletions = []

families = phase_data.get_families()
for family in families:
	print(family)

	individuals = family['individuals']
	if phase_data.is_standard_family(family['family']) and family['is_fully_phased']:
		chroms, starts, ends, deletions, mat_phases, pat_phases, is_hts = pull_phase_data_into_arrays(family['family'])

		for chrom in set(chroms):
			is_chrom = np.array([x==chrom for x in chroms])
			interval_lengths_chrom = (ends-starts)[is_chrom]
			
			# pull inherited deletions
			for anc in range(4):
				is_mat = anc==0 or anc==1
				is_pat = anc==2 or anc==3

				for opt_start_index, start_index, end_index, opt_end_index in pull_deletion_indices(deletions[anc, is_chrom]):
					assert np.all(deletions[anc, is_chrom][start_index:end_index]==0)
					assert np.all(deletions[anc, is_chrom][opt_start_index:opt_end_index]<1)

					is_del_hts = np.sum(interval_lengths_chrom[start_index:end_index][is_hts[is_chrom][start_index:end_index]])/np.sum(interval_lengths_chrom[start_index:end_index]) > 0.9

					trans, notrans = [], []
					for i, child in enumerate(individuals):
						if i>=2 and is_mat:
							fraction_del_inherited = np.sum(interval_lengths_chrom[start_index:end_index][mat_phases[i, is_chrom][start_index:end_index]==anc])/np.sum(interval_lengths_chrom[start_index:end_index])
							if fraction_del_inherited>0.9:
								trans.append(child)
							else:
								notrans.append(child)
					

					start_pos, end_pos = starts[is_chrom][start_index], starts[is_chrom][end_index]
					opt_start_pos, opt_end_pos = starts[is_chrom][opt_start_index], starts[is_chrom][opt_end_index]
					length = end_pos - start_pos + 1

					assert start_pos <= end_pos
					assert opt_start_pos <= start_pos
					assert end_pos <= opt_end_pos
					assert length > 0

					all_deletions.append(Deletion(family['family'], chrom,
								start_pos, end_pos, length,
								opt_start_pos, opt_end_pos, tuple(trans), tuple(notrans),
								is_mat, is_pat, is_del_hts,
								individuals[0] if is_mat else individuals[1]))



print('deletions', len(all_deletions))

# write to json
with open('%s/deletions.json' % phase_data.phase_dir, 'w+') as f:
	json.dump([d._asdict() for d in all_deletions], f, indent=4, cls=NumpyEncoder)

# # create collections
# class DeletionCollection:
# 	def __init__(self, deletion, matches):
# 		self.deletion = deletion
# 		self.matches = matches

# collections = []
	    
# starts = np.array([d.opt_start_pos for d in deletions])
# stops = np.array([d.opt_end_pos for d in deletions])

# ordered_start_indices = np.argsort(starts)
# ordered_starts = starts[ordered_start_indices]
# ordered_stop_indices = np.argsort(stops)
# ordered_stops = stops[ordered_stop_indices]
	        
# insert_starts_in_stops = np.searchsorted(ordered_stops, starts)
# insert_stops_in_starts = np.searchsorted(ordered_starts, stops, side='right')
	        
# indices = np.ones((len(deletions),), dtype=bool)

# for del_index, main_d in enumerate(deletions):
# 	indices[:] = True
# 	indices[ordered_stop_indices[:insert_starts_in_stops[del_index]]] = False
# 	indices[ordered_start_indices[insert_stops_in_starts[del_index]:]] = False

# 	collections.append(DeletionCollection(main_d, [deletions[j] for j in np.where(indices)[0]]))
# print('collections', len(collections))

# ## prune deletions
# for c in collections:
# 	# we know all deletions within a collection overlap at least a little bit

# 	# this method focuses on finding deletions which overlap by 50%
# 	lengths = np.array([d.length for d in c.matches])
# 	overlaps = np.array([min(d.end_pos, c.deletion.end_pos)-max(d.start_pos, c.deletion.start_pos)+1 for d in c.matches])
# 	c.matches = set([c.matches[j] for j in np.where((overlaps >= 0.5*lengths) & (overlaps >= 0.5*c.deletion.length))[0]])

# print('deletions pruned')

# # prune collections (get rid of collections that are identical to other collections)
# deletion_to_index = dict([(x.deletion, i) for i, x in enumerate(collections)])
	    
# for c in collections:
# 	if c is not None:
# 		for d in c.matches:
# 			index = deletion_to_index[d]
# 			if (c.deletion != d) and (collections[index] is not None) and (c.matches == collections[index].matches):
# 				collections[index] = None
# print('collections pruned, removed %d of %d' % (len([x for x in collections if x is None]), len(collections)))
# collections = [x for x in collections if x is not None]

# # write to json
# json_collections = list()
# for collection in collections:
# 	json_collections.append({
# 		'deletion': collection.deletion._asdict(),
# 		'matches': [m._asdict() for m in collection.matches]
# 	})
# with open('%s/collections.json' % dataset_name, 'w+') as f:
# 	json.dump(json_collections, f, indent=4, cls=NumpyEncoder)
    



