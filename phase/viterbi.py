import numpy as np
import time

def viterbi_forward_sweep(family_genotypes, family_snp_positions, mult_factor, states, transition_matrix, loss):
		
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	v_cost = np.zeros((states.num_states, n), dtype=float)

	# first step, break symmetry
	# we enforce that the chromosome starts with no deletions and a hard to sequence region
	# also, no de novo deletions
	pos_gen = tuple(family_genotypes[:, 0])
	v_cost[:, 0] = mult_factor[0]*loss(pos_gen)

	ok_start = np.array([not x.has_deletion() and x.is_hard_to_sequence() for x in states])
	v_cost[~ok_start, 0] = np.inf

	# next steps
	for j in range(1, n): 
		pos_gen = tuple(family_genotypes[:, j])
		v_cost[:, j] = np.min(v_cost[transition_matrix.transitions, j-1] + transition_matrix.costs, axis=1) + mult_factor[j]*loss(pos_gen)

	print('Forward sweep complete', time.time()-prev_time, 'sec') 

	return v_cost

def merge_paths(paths, states):
	# combine path states a single state (unknown values represented with -1)
	merged_state = -np.ones((states._states.shape[1],), dtype=np.int8)
	if paths.shape[0] == 1:
		merged_state = states[paths[0]]
	else:
		path_states = states[paths]
		known_indices = np.all(path_states == path_states[0, :], axis=0)
		merged_state[known_indices] = path_states[0, known_indices]
	return merged_state

def viterbi_backward_sweep(v_cost, states, transition_matrix):

	# backward sweep
	prev_time = time.time()
	n = v_cost.shape[1]
	final_states = -np.ones((states._states.shape[1], n), dtype=int)
	
	# choose best paths
	# we enforce that the chromosome ends with no deletions and a hard to sequence region
	num_forks = 0
	ok_end = np.array([not x.has_deletion() and x.is_hard_to_sequence() for x in states])
	min_value = np.min(v_cost[ok_end, -1])
	paths = np.where(np.isclose(v_cost[:, -1], min_value) & ok_end)[0]
	print('Num solutions', paths.shape, min_value, states[paths])

	final_states[:, -1] = merge_paths(paths, states)
	num_forks += (paths.shape[0] > 1)

	# now work backwards
	for j in reversed(range(n-1)):
		# traceback
		total_cost = v_cost[transition_matrix.transitions[paths, :], j] + transition_matrix.costs[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = transition_matrix.transitions[k, np.where(np.isclose(total_cost[i, :], min_value[i]))[0]]	
			new_paths.update(min_indices.tolist())
		
		paths = np.asarray(list(new_paths), dtype=int)
		final_states[:, j] = merge_paths(paths, states)
		num_forks += (paths.shape[0] > 1)

	assert np.all(final_states[[0, 1, 2, 3, -1], 0] == 1)
	assert np.all(final_states[[0, 1, 2, 3, -1], -1] == 1)

	print('Num positions in fork', num_forks)
	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states


