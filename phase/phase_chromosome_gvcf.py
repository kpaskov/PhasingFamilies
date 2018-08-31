import sys
import time
from os import listdir
import gzip

from collections import Counter, Hashable, defaultdict
from itertools import chain, product

import numpy as np
from scipy import sparse
import random


# Run locally with python3 phase/phase_chromosome_gvcf.py 22 split_gen_gvcf/chr.22.AU1274.AU1274202.AU1274201.gen.npy phased_gvcf

chrom = sys.argv[1]
input_file = sys.argv[2]
out_dir = sys.argv[3]

# pull data from .npy
familyreads = np.load(input_file)

# randomly permute children
m = familyreads.shape[0]
permutation_order = [0, 1]+random.sample(range(2, m), m-2)
familyreads = familyreads[permutation_order, :, :]

# inheritance states
#
# for parents:
# (0, 0) -> deletion on parental1 and parental2
# (0, 1) -> deletion on parental1
# (0, 2) -> deletion on parental1 and duplication on parental2
# (1, 0) -> deletion on parental2
# (1, 1) -> normal
# (1, 2) -> duplication on parental2
# (2, 0) -> duplication on parental1 and deletion on parental2
# (2, 1) -> duplication on parental1
# (2, 2) -> duplication on parental1 and parental2
# 
# for children:
# (0, 0) -> m1p1
# (0, 1) -> m1p2
# (1, 0) -> m2p1
# (1, 1) -> m2p2

if m >= 5:
	inheritance_states = np.array(list(product(*(([[0, 1, 2]]*4)+([[0, 1]]*(2*(m-2)))))), dtype=np.int8)
else:
	inheritance_states = np.array([x for x in product(*([[0, 1, 2]]*4)+([[0, 1]]*(2*(m-2)))) if x[4] == 0 and x[5] == 0], dtype=np.int8)
state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states)])
p, state_len = inheritance_states.shape
print('inheritance states', inheritance_states.shape)

# transition matrix
# only allow one shift at a time
shift_costs = [50]*4 + [500]*(2*(m-2))

transitions = [[] for i in range(p)]
transition_costs = [[] for i in range(p)]
for i, state in enumerate(inheritance_states):
	for delstate in list(product(*[[0, 1, 2]]*4)):
		new_state = tuple(delstate) + tuple(state[4:])
		new_index = state_to_index[new_state]
		transitions[i].append(new_index)
		transition_costs[i].append(sum([shift_costs[j] for j, (old_s, new_s) in enumerate(zip(state[:4], delstate)) if old_s != new_s]))

	# allow a single recombination event
	for j in range(4, inheritance_states.shape[1]):
		new_state = tuple(1-x if k == j else x for k, x in enumerate(state))
		if new_state in state_to_index:
			new_index = state_to_index[new_state]
			transitions[i].append(new_index)
			transition_costs[i].append(shift_costs[j])
            
transitions = np.array(transitions)
transition_costs = np.array(transition_costs)

starting_state = (1, 1, 1, 1, 0, 0)
zero_transition_costs = np.zeros((p,), dtype=int)
for i, ss in enumerate(starting_state):
	zero_transition_costs[inheritance_states[:, i] != ss] += shift_costs[i]
print('transitions', transitions.shape)

# condense repeated genotypes
rep_indices = np.where(np.any(family_genotypes[:, 1:]!=family_genotypes[:, :-1], axis=0))[0]
mult_factor = [rep_indices[0]+1] + (rep_indices[1:]-rep_indices[:-1]).tolist() + [family_genotypes.shape[1]-rep_indices[-1]-1]
family_snp_positions = np.zeros((rep_indices.shape[0]+1, 2), dtype=int)
family_snp_positions[1:, 0] = snp_positions[(rep_indices+1)]
family_snp_positions[0, 0] = snp_positions[0]
family_snp_positions[:-1, 1] = snp_positions[rep_indices]
family_snp_positions[-1, 1] = snp_positions[-1]
rep_indices = rep_indices.tolist()
rep_indices.append(family_genotypes.shape[1]-1)
n = len(rep_indices)

# viterbi
v_cost = np.zeros((p, n), dtype=int)

# forward sweep
prev_time = time.time()

# first step, break symmetry
# we enforce that the chromosome starts with child1 (0, 0) and no deletions or duplications
pos_gen = tuple(family_genotypes[:, 0])
loss = calculate_loss(pos_gen).astype(int)
#loss = losses[:, genotype_to_index[pos_gen]].astype(int)
v_cost[:, 0] = mult_factor[0]*loss + zero_transition_costs

# next steps
for j in range(1, n): 
	pos_gen = tuple(family_genotypes[:, rep_indices[j]])
	loss = calculate_loss(pos_gen).astype(int)
	#loss = losses[:, genotype_to_index[pos_gen]].astype(int)
	v_cost[:, j] = np.min(v_cost[transitions, j-1] + transition_costs, axis=1) + mult_factor[j]*loss

print('Forward sweep complete', time.time()-prev_time, 'sec') 

# write header to file
famf.write('%s\t%s\n' % ('.'.join(fkey), '\t'.join(inds)))
famf.flush()

# backward sweep
prev_time = time.time()
final_states = -np.ones((state_len, n), dtype=int)

# choose best paths
# we enforce that the chromosome ends with no deletions
num_forks = 0
no_delstates = np.sum(inheritance_states[:, :4] == 1, axis=1)==4  
min_value = np.min(v_cost[no_delstates, -1])
paths = np.where((v_cost[:, -1]==min_value) & no_delstates)[0]
print('Num solutions', paths.shape, inheritance_states[paths, :])

# combine path states into a single state (unknown values represented with -1)
if paths.shape[0] == 1:
	final_states[:, -1] = inheritance_states[paths[0], :]
else:
	num_forks += 1
	path_states = inheritance_states[paths, :]
	known_indices = np.all(path_states == path_states[0, :], axis=0)
	final_states[known_indices, -1] = path_states[0, known_indices]

# now work backwards
for j in reversed(range(n-1)):

	# traceback
	total_cost = v_cost[transitions[paths, :], j] + transition_costs[paths, :]
	min_value = np.min(total_cost, axis=1)
	new_paths = set()
	for i, k in enumerate(paths):
		min_indices = transitions[k, np.where(total_cost[i, :] == min_value[i])[0]]	
		new_paths.update(min_indices.tolist())
	paths = np.asarray(list(new_paths), dtype=int)

	# combine path states a single state (unknown values represented with -1)
	if paths.shape[0] == 1:
		final_states[:, j] = inheritance_states[paths[0], :]
	else:
		num_forks += 1
		path_states = inheritance_states[paths, :]
		known_indices = np.all(path_states == path_states[0, :], axis=0)
		final_states[known_indices, j] = path_states[0, known_indices]

print('Num positions in fork', num_forks)
print('Backward sweep complete', time.time()-prev_time, 'sec') 

# if a parental chromosome isn't inherited, then we don't know if it has a deletion
maternal_indices = range(4, state_len, 2)
paternal_indices = range(5, state_len, 2)

final_states[0, np.all(final_states[maternal_indices, :]!=0, axis=0)] = -1
final_states[1, np.all(final_states[maternal_indices, :]!=1, axis=0)] = -1
final_states[2, np.all(final_states[paternal_indices, :]!=0, axis=0)] = -1
final_states[3, np.all(final_states[paternal_indices, :]!=1, axis=0)] = -1

# write to file
with open('%s/chr.%s.familysize.%s.phased.txt' % (out_dir, chrom, m), 'w+') as statef:
	# write headers
	statef.write('\t'.join(['family_id', 'state_id', 'm1_state', 'm2_state', 'p1_state', 'p2_state',
		'\t'.join(['child%d_%s_state' % ((i+1), c) for i, c in product(range(m-2), ['m', 'p'])]),
		'start_pos', 'end_pos', 'start_family_index', 'end_family_index' 'pos_length', 'family_index_length']) + '\n')

	change_indices = [-1] + np.where(np.any(final_states[:, 1:]!=final_states[:, :-1], axis=0))[0].tolist()
	for j in range(1, len(change_indices)):
		s_start, s_end = change_indices[j-1]+1, change_indices[j]
		#assert np.all(final_states[:, s_start] == final_states[:, s_end])
		statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
					'.'.join(fkey), 
					'\t'.join(map(str, final_states[:, s_start])), 
					family_snp_positions[s_start, 0], family_snp_positions[s_end, 1],
					s_start, s_end, 
					family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0]+1, 
					s_end-s_start+1))

	# last entry
	s_start, s_end = change_indices[-1]+1, family_snp_positions.shape[0]-1
	#assert np.all(final_states[:, s_start] == final_states[:, s_end])
	statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
				'.'.join(fkey), 
				'\t'.join(map(str, final_states[:, s_start])), 
				family_snp_positions[s_start, 0], family_snp_positions[s_end, 1],
				s_start, s_end, 
				family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0]+1, 
				s_end-s_start+1))
	statef.flush()	

	print('Write to file complete')
