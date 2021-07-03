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

			positions, states = [], []
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

					if chrom == 'X':
						positions.append(start_pos)
						states.append(state)

			if np.all([x.endswith('_del') for x in header[1:5]]) and not header[5].endswith('_del') and len(states)>0:
				# pull only "simple" families with mom, dad, and children
				families.add(family_key)
				
				positions = np.array(positions)
				states = np.array(states)
				#states[states[:, -1]==1] = -1
			
				individuals.update(inds)

				interval_lengths = positions[1:]-positions[:-1]

				assert np.all((states[:, :4]>=-1) & (states[:, :4]<=1))
					
				# pull inherited deletions
				for anc in range(4):
					is_mat = anc==0 or anc==1
					is_pat = anc==2 or anc==3

					for opt_start_index, start_index, end_index, opt_end_index in pull_deletion_indices(states[:, anc]):
						assert np.all(states[start_index:end_index, anc]==0)
						assert np.all(states[opt_start_index:opt_end_index, anc]<1)
							
						if is_mat:
							parental_indices = np.arange(4, 4+(2*family_size), 2)
						if is_pat:
							parental_indices = np.arange(5, 4+(2*family_size), 2)

						majority_parental_inheritance = None
						for parental_inheritance_option in np.unique(states[start_index:end_index, :][:, parental_indices], axis=0):
							has_option = np.all(states[start_index:end_index, :][:, parental_indices] == np.tile(parental_inheritance_option, ((end_index-start_index), 1)), axis=1)
							if np.sum(interval_lengths[start_index:end_index][has_option])>0.9*np.sum(interval_lengths[start_index:end_index]):
								majority_parental_inheritance = parental_inheritance_option

						#has_hts = states[start_index:end_index, -1]!=0
							
						if majority_parental_inheritance is not None and np.all(majority_parental_inheritance!=-1):
							start_pos, end_pos = positions[start_index], positions[end_index]
							opt_start_pos, opt_end_pos = positions[opt_start_index], positions[opt_end_index]
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

					
		except StopIteration:
			pass
		#except ValueError:
		#	print('Value Error', pieces)


deletions = sorted(deletions, key=lambda x: x.start_pos)
print('deletions', len(deletions), 
	'inherited', len([x for x in deletions if x.is_inherited]),
	'denovo', len([x for x in deletions if x.is_denovo]))

# write to json
json_deletions = list()
for deletion in deletions:
	json_deletions.append(deletion._asdict())

with open('%s/Xdeletions.json' % phase_dir, 'w+') as f:
	json.dump(json_deletions, f, indent=4)
