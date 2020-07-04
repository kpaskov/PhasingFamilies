import numpy as np
import time
import random

def viterbi_forward_sweep(family_genotypes, family_snp_positions, mult_factor, states, transition_matrix, loss):
		
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	v_cost = np.zeros((states.num_states, n), dtype=float)

	v_cost[:, 0] = mult_factor[0]*loss(family_genotypes[:, 0])

	# we enforce that the chromosome starts with no deletions
	ok_start = np.array([states.is_ok_start(x) for x in states])
	v_cost[~ok_start, 0] = np.inf

	# next steps
	for j in range(1, n): 
		v_cost[:, j] = np.min(v_cost[transition_matrix.transitions, j-1] + transition_matrix.costs, axis=1) + mult_factor[j]*loss(family_genotypes[:, j])

	print('Forward sweep complete', time.time()-prev_time, 'sec') 

	return v_cost

def merge_paths(paths, states):
	# combine path states into a single state (unknown values represented with -1)
	path_states = states.get_full_states(paths)
	return ((path_states[0, :]+1)*np.all(np.equal(path_states, path_states[0, :]), axis=0)) - 1

def viterbi_backward_sweep(v_cost, states, transition_matrix):

	# backward sweep
	prev_time = time.time()
	n = v_cost.shape[1]
	final_states = -np.ones((states.full_state_length, n), dtype=np.int8)
	
	# choose best paths
	# we enforce that the chromosome ends with no deletions
	num_forks = 0
	ok_end = np.array([states.is_ok_end(x) for x in states])
	min_value = np.min(v_cost[ok_end, -1])
	paths = np.where(np.isclose(v_cost[:, -1], min_value, rtol=0, atol=0.1) & ok_end)[0]
	print('Num solutions', paths.shape, min_value, states[paths])

	if paths.shape[0]>1:
		paths = [random.choice(paths)]

	final_states[:, -1] = merge_paths(paths, states)
	num_forks += (paths.shape[0] > 1)

	# now work backwards
	for j in reversed(range(n-1)):
		# traceback
		total_cost = v_cost[transition_matrix.transitions[paths, :], j] + transition_matrix.costs[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = transition_matrix.transitions[k, np.isclose(total_cost[i, :], min_value[i], rtol=0, atol=0.1)]	
			new_paths.update(min_indices.tolist())
		
		paths = list(new_paths)
		final_states[:, j] = merge_paths(paths, states)
		num_forks += (len(paths) > 1)

	print('Num forks', num_forks)
	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states


