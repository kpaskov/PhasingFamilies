from collections import defaultdict, namedtuple
import sys
import json
import numpy as np
import math
from os import listdir
from enum import Enum

phase_dir = sys.argv[1] #'../phased_ihart'

# read in deletions
Deletion = namedtuple('Deletion', ['family', 'chrom', 'start_pos', 'end_pos', 'length',
	'opt_start_pos', 'opt_end_pos', 
	'trans', 'notrans', 'family_size', 'is_mat', 'is_pat', 'mother', 'father', 'is_denovo', 'is_inherited'])

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

deletions = []
individuals = set()
families = set()

for filename in listdir(phase_dir):
	if filename.endswith('.phased.txt'):
		try:
			family_key = filename[:-11]
			print(family_key)

			chroms, positions, states = [], [], []
			with open('%s/%s' % (phase_dir, filename), 'r')  as f:
				header = next(f).strip().split('\t')
				inds = [header[i][:-4] for i in range(5, len(header)-3, 2)]
				family_size = len(inds)

				for line in f:
					pieces = line.strip().split('\t')
					chrom = pieces[0][3:]
					start_pos, end_pos = [int(x) for x in pieces[-2:]]
					state = np.array([int(x) for x in pieces[1:-2]])
				
					assert end_pos >= start_pos

					if chrom != 'X':
						chroms.append(chrom)
						positions.append(start_pos)
						states.append(state)

			if np.all([x.endswith('_del') for x in header[1:5]]) and not header[5].endswith('_del') and len(set(chroms))>=22:
				# pull only "simple" families with mom, dad, and children
				families.add(family_key)
				
				positions = np.array(positions)
				states = np.array(states)
				#states[states[:, -1]==1] = -1
			
				individuals.update(inds)

				for chrom in set(chroms):
					chrom_states = states[[c==chrom for c in chroms], :].copy()
					chrom_positions = positions[[c==chrom for c in chroms]].copy()
					interval_lengths = chrom_positions[1:]-chrom_positions[:-1]

					assert np.all((chrom_states[:, :4]>=-1) & (chrom_states[:, :4]<=1))
					
					# pull inherited deletions
					for anc in range(4):
						is_mat = anc==0 or anc==1
						is_pat = anc==2 or anc==3

						for opt_start_index, start_index, end_index, opt_end_index in pull_deletion_indices(chrom_states[:, anc]):
							assert np.all(chrom_states[start_index:end_index, anc]==0)
							assert np.all(chrom_states[opt_start_index:opt_end_index, anc]<1)
							
							if is_mat:
								parental_indices = np.arange(4, 4+(2*family_size), 2)
							if is_pat:
								parental_indices = np.arange(5, 4+(2*family_size), 2)

							majority_parental_inheritance = None
							for parental_inheritance_option in np.unique(chrom_states[start_index:end_index, :][:, parental_indices], axis=0):
								has_option = np.all(chrom_states[start_index:end_index, :][:, parental_indices] == np.tile(parental_inheritance_option, ((end_index-start_index), 1)), axis=1)
								if np.sum(interval_lengths[start_index:end_index][has_option])>0.9*np.sum(interval_lengths[start_index:end_index]):
									majority_parental_inheritance = parental_inheritance_option

							#has_hts = chrom_states[start_index:end_index, -1]!=0
							
							if majority_parental_inheritance is not None and np.all(majority_parental_inheritance!=-1):
								start_pos, end_pos = chrom_positions[start_index], chrom_positions[end_index]
								opt_start_pos, opt_end_pos = chrom_positions[opt_start_index], chrom_positions[opt_end_index]
								length = int(end_pos - start_pos + 1)

								assert start_pos <= end_pos
								assert opt_start_pos <= start_pos
								assert end_pos <= opt_end_pos

								# children
								trans, notrans = [], []
								for child, par_inh in zip(inds[2:], majority_parental_inheritance[2:]):
									if par_inh==anc:
										trans.append(child)
									else:
										notrans.append(child)

								if len(trans) + len(notrans) == family_size-2:
									deletions.append(Deletion(family_key, chrom,
										int(start_pos), int(end_pos), length,
										int(opt_start_pos), int(opt_end_pos), tuple(trans), tuple(notrans),
										len(inds), is_mat, is_pat,
										inds[0], inds[1], 
										False, True))

					# pull de novo deletions
					for child_index in range(2, len(inds)):

						mat_state = chrom_states[:, 4 + 2*len(inds) + 2*child_index]
						pat_state = chrom_states[:, 5 + 2*len(inds) + 2*child_index]

						# maternal de novos
						for opt_start_index, start_index, end_index, opt_end_index in pull_deletion_indices(mat_state):
							assert np.all(mat_state[start_index:end_index]==0)
							assert np.all(mat_state[opt_start_index:opt_end_index]<1)
							
							parental_indices = np.arange(4, 4+(2*family_size), 2)

							has_recomb = not np.all(chrom_states[start_index:end_index, :][:, parental_indices] == np.tile(chrom_states[start_index, parental_indices], ((end_index-start_index), 1)))
							has_unknown_phase = np.any(chrom_states[start_index, parental_indices]==-1)
							has_hts = np.any(chrom_states[start_index:end_index, -1]!=0)
							
							if (not has_recomb) and (not has_unknown_phase) and (not has_hts):
								start_pos, end_pos = chrom_positions[start_index], chrom_positions[end_index]
								opt_start_pos, opt_end_pos = chrom_positions[opt_start_index], chrom_positions[opt_end_index]
								length = int(end_pos - start_pos + 1)

								assert start_pos <= end_pos
								assert opt_start_pos <= start_pos
								assert end_pos <= opt_end_pos

								deletions.append(Deletion(family_key, chrom,
											int(start_pos), int(end_pos), length,
											int(opt_start_pos), int(opt_end_pos), (inds[child_index],), tuple([x for x in inds[2:] if x!=inds[child_index]]),
											len(inds), True, False,
											inds[0], inds[1], 
											True, False))

						# paternal de novos
						for opt_start_index, start_index, end_index, opt_end_index in pull_deletion_indices(pat_state):
							assert np.all(pat_state[start_index:end_index]==0)
							assert np.all(pat_state[opt_start_index:opt_end_index]<1)
							
							parental_indices = np.arange(5, 4+(2*family_size), 2)

							has_recomb = not np.all(chrom_states[start_index:end_index, :][:, parental_indices] == np.tile(chrom_states[start_index, parental_indices], ((end_index-start_index), 1)))
							has_unknown_phase = np.any(chrom_states[start_index, parental_indices]==-1)
							has_hts = np.any(chrom_states[start_index:end_index, -1]!=0)
							
							if (not has_recomb) and (not has_unknown_phase) and (not has_hts):
								start_pos, end_pos = chrom_positions[start_index], chrom_positions[end_index]
								opt_start_pos, opt_end_pos = chrom_positions[opt_start_index], chrom_positions[opt_end_index]
								length = int(end_pos - start_pos + 1)

								assert start_pos <= end_pos
								assert opt_start_pos <= start_pos
								assert end_pos <= opt_end_pos

								deletions.append(Deletion(family_key, chrom,
											int(start_pos), int(end_pos), length,
											int(opt_start_pos), int(opt_end_pos), (inds[child_index],), tuple([x for x in inds[2:] if x!=inds[child_index]]),
											len(inds), False, True,
											inds[0], inds[1], 
											True, False))

					
		except StopIteration:
			pass
		#except ValueError:
		#	print('Value Error', pieces)

with open('%s/individuals.json' % phase_dir, 'w+') as f:
	json.dump(sorted(individuals), f, indent=4)
with open('%s/families.json' % phase_dir, 'w+') as f:
	json.dump(sorted(families), f, indent=4)

deletions = sorted(deletions, key=lambda x: x.start_pos)
print('deletions', len(deletions), 
	'inherited', len([x for x in deletions if x.is_inherited]),
	'denovo', len([x for x in deletions if x.is_denovo]))

# write to json
json_deletions = list()
for deletion in deletions:
	json_deletions.append(deletion._asdict())

with open('%s/deletions.json' % phase_dir, 'w+') as f:
	json.dump(json_deletions, f, indent=4)

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
# with open('%s/collections.json' % phase_dir, 'w+') as f:
# 	json.dump(json_collections, f, indent=4)
    



