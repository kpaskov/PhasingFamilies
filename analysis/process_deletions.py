from collections import defaultdict, namedtuple
import sys
import json
import numpy as np
import math
from os import listdir
from enum import Enum

phase_dir = sys.argv[1] #'../phased_ihart'

# read in deletions
CNV = namedtuple('CNV', ['family', 'chrom', 'start_pos', 'end_pos', 'length',
	'opt_start_pos', 'opt_end_pos', 
	'trans', 'notrans', 'family_size', 'is_mat', 'is_pat', 'mother', 'father', 'is_deletion', 'is_duplication', 'is_denovo', 'is_inherited'])


deletions, duplications = [], []
individuals = set()
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
			
			#if len(set(chroms))==22:
			positions = np.array(positions)
			states = np.array(states)
			states[states[:, -1]==1] = -1
		
			individuals.update(inds)

			for chrom in set(chroms):
				chrom_states = states[[c==chrom for c in chroms], :].copy()
				chrom_positions = positions[[c==chrom for c in chroms]].copy()

				# deal with haplotype linking
				chrom_states[chrom_states[:, 1]==3, 1] = chrom_states[chrom_states[:, 1]==3, 0]
				chrom_states[chrom_states[:, 2]==3, 2] = chrom_states[chrom_states[:, 2]==3, 0]
				chrom_states[chrom_states[:, 2]==4, 2] = chrom_states[chrom_states[:, 2]==4, 1]
				chrom_states[chrom_states[:, 3]==3, 3] = chrom_states[chrom_states[:, 3]==3, 0]
				chrom_states[chrom_states[:, 3]==4, 3] = chrom_states[chrom_states[:, 3]==4, 1]
				chrom_states[chrom_states[:, 3]==5, 3] = chrom_states[chrom_states[:, 3]==5, 2]

				assert np.all((chrom_states[:, :4]>=-1) & (chrom_states[:, :4]<=2))
				
				# pull inherited deletions
				for anc in range(4):
					is_mat = anc==0 or anc==1
					is_pat = anc==2 or anc==3

					nohap_anc_state = chrom_states[:, anc]

					anc_state_change_indices = np.where(nohap_anc_state[:-1] != nohap_anc_state[1:])[0]
					anc_states = np.zeros((len(anc_state_change_indices)+1), dtype=int)
					anc_states[:-1] = nohap_anc_state[anc_state_change_indices]
					anc_states[-1] = nohap_anc_state[-1]
					anc_indices = [0] + (anc_state_change_indices+1).tolist() + [len(nohap_anc_state)-1]

					for anc_index in np.where(anc_states==0)[0]:
						# check that inheritance state is known
						start_index, end_index = anc_indices[anc_index], anc_indices[anc_index+1]
						inh_state = chrom_states[start_index, :]
						assert np.all(chrom_states[start_index:end_index, anc]==0)
						assert chrom_states[start_index:end_index, :][:, np.arange(4, 4+(2*family_size))].shape == np.tile(inh_state[np.arange(4, 4+(2*family_size))], ((end_index-start_index), 1)).shape
						has_recomb = not np.all(chrom_states[start_index:end_index, :][:, np.arange(4, 4+(2*family_size))] == np.tile(inh_state[np.arange(4, 4+(2*family_size))], ((end_index-start_index), 1)))
						has_unknown_phase = np.any(inh_state[np.arange(4, 4+(2*family_size))]==-1)
						if (not has_recomb) and (not has_unknown_phase):
							start_pos, end_pos = chrom_positions[start_index], chrom_positions[end_index]
							length = int(end_pos - start_pos + 1)

							if start_index == 0 or anc_states[anc_index-1] == 1:
								opt_start_index = start_index
							else:
								opt_start_index = anc_indices[anc_index-1]

							if end_index == len(nohap_anc_state)-1 or anc_states[anc_index+1] == 1:
								opt_end_index = end_index
							else:
								opt_end_index = anc_indices[anc_index+2]
							opt_start_pos, opt_end_pos = chrom_positions[opt_start_index], chrom_positions[opt_end_index]

							assert start_pos <= end_pos
							assert opt_start_pos <= start_pos
							assert end_pos <= opt_end_pos

							# children
							trans, notrans = [], []
							for k, child in zip(range(2, family_size), inds[2:]):
								mom_s, dad_s = inh_state[(4+(2*k)):(6+(2*k))]

								if is_mat:
									assert mom_s != -1
									if anc==mom_s:
										trans.append(child)
									else:
										notrans.append(child)
								if is_pat:
									assert dad_s != -1
									if anc==dad_s:
										trans.append(child)
									else:
										notrans.append(child)

							if len(trans) + len(notrans) == family_size-2:
								#print(chrom, opt_start_pos, start_pos, end_pos, opt_end_pos)
								deletions.append(CNV(family_key, chrom,
									int(start_pos), int(end_pos), length,
									int(opt_start_pos), int(opt_end_pos), tuple(trans), tuple(notrans),
									len(inds), is_mat, is_pat,
									inds[0], inds[1], 
									True, False, 
									False, True))

				# # pull de novo deletions
				# for child_index in range(2, len(inds)):

				# 	mat_state = chrom_states[:, 4 + 2*len(inds) + 2*child_index]
				# 	pat_state = chrom_states[:, 5 + 2*len(inds) + 2*child_index]

				# 	mat_state_change_indices = np.where(mat_state[:-1] != mat_state[1:])[0]
				# 	mat_states = np.zeros((len(mat_state_change_indices)+1), dtype=int)
				# 	mat_states[:-1] = mat_state[mat_state_change_indices]
				# 	mat_states[-1] = mat_state[-1]
				# 	mat_indices = [0] + (mat_state_change_indices+1).tolist() + [len(mat_state)-1]

				# 	for mat_index in np.where(mat_states==1)[0]:
				# 		start_index, end_index = mat_indices[mat_index], mat_indices[mat_index+1]
				# 		start_pos, end_pos = chrom_positions[start_index], chrom_positions[end_index]
				# 		length = int(end_pos - start_pos + 1)

				# 		if start_index == 0 or mat_states[mat_index-1] == 1:
				# 			opt_start_index = start_index
				# 		else:
				# 			opt_start_index = mat_indices[mat_index-1]

				# 		if end_index == len(mat_state)-1 or mat_states[mat_index+1] == 1:
				# 			opt_end_index = end_index
				# 		else:
				# 			opt_end_index = mat_indices[mat_index+2]
				# 		opt_start_pos, opt_end_pos = chrom_positions[opt_start_index], chrom_positions[opt_end_index]

				# 		print(opt_start_pos, start_pos, end_pos, opt_end_pos)
				# 		assert start_pos <= end_pos
				# 		assert opt_start_pos <= start_pos
				# 		assert end_pos <= opt_end_pos

				# 		deletions.append(CNV(family_key, chrom,
				# 					int(start_pos), int(end_pos), length,
				# 					int(opt_start_pos), int(opt_end_pos), (inds[child_index],), tuple([x for x in inds[2:] if x!=inds[child_index]]),
				# 					len(inds), True, False,
				# 					inds[0], inds[1], 
				# 					True, False,
				# 					True, False))

				# 	pat_state_change_indices = np.where(pat_state[:-1] != pat_state[1:])[0]
				# 	pat_states = np.zeros((len(pat_state_change_indices)+1), dtype=int)
				# 	pat_states[:-1] = mat_state[pat_state_change_indices]
				# 	pat_states[-1] = pat_state[-1]
				# 	pat_indices = [0] + (pat_state_change_indices+1).tolist() + [len(pat_state)-1]

				# 	for pat_index in np.where(pat_states==1)[0]:
				# 		start_index, end_index = pat_indices[pat_index], pat_indices[pat_index+1]
				# 		start_pos, end_pos = chrom_positions[start_index], chrom_positions[end_index]
				# 		length = int(end_pos - start_pos + 1)

				# 		if start_index == 0 or pat_states[pat_index-1] == 1:
				# 			opt_start_index = start_index
				# 		else:
				# 			opt_start_index = pat_indices[mat_index-1]

				# 		if end_index == len(pat_state)-1 or pat_states[pat_index+1] == 1:
				# 			opt_end_index = end_index
				# 		else:
				# 			opt_end_index = pat_indices[pat_index+2]
				# 		opt_start_pos, opt_end_pos = chrom_positions[opt_start_index], chrom_positions[opt_end_index]

				# 		assert start_pos <= end_pos
				# 		assert opt_start_pos <= start_pos
				# 		assert end_pos <= opt_end_pos

				# 		deletions.append(CNV(family_key, chrom,
				# 					int(start_pos), int(end_pos), length,
				# 					int(opt_start_pos), int(opt_end_pos), (inds[child_index],), tuple([x for x in inds[2:] if x!=inds[child_index]]),
				# 					len(inds), False, True,
				# 					inds[0], inds[1], 
				# 					True, False,
				# 					True, False))

					
		except StopIteration:
			pass
		except ValueError:
			print('Value Error', pieces)

with open('%s/individuals.json' % phase_dir, 'w+') as f:
	json.dump(sorted(individuals), f, indent=4)


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
    



