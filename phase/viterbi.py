import numpy as np
import time

def viterbi_forward_sweep_autosomes(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, loss):
		
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	p, state_len = inheritance_states.p, inheritance_states.state_len
	v_cost = np.zeros((p, n), dtype=float)

	# first step, break symmetry
	# we enforce that the chromosome starts with no deletions and a hard to sequence region
	# also, no de novo deletions
	pos_gen = tuple(family_genotypes[:, 0])
	v_cost[:, 0] = mult_factor[0]*loss(pos_gen)

	no_delstates = np.all(inheritance_states[:, [0, 1, 2, 3, -1]]==1, axis=1) & np.all(inheritance_states[:, np.arange(2*inheritance_states.m, 2*inheritance_states.m + 2*(inheritance_states.m - 2))]==0, axis=1)
	v_cost[~no_delstates, 0] = np.inf

	# next steps
	for j in range(1, n): 
		pos_gen = tuple(family_genotypes[:, j])
		v_cost[:, j] = np.min(v_cost[transition_matrix.transitions, j-1] + transition_matrix.costs, axis=1) + mult_factor[j]*loss(pos_gen)

	print('Forward sweep complete', time.time()-prev_time, 'sec') 

	return v_cost

def merge_paths(paths, inheritance_states):
	# combine path states a single state (unknown values represented with -1)
	merged_state = -np.ones((inheritance_states.state_len,), dtype=np.int8)
	if paths.shape[0] == 1:
		merged_state = inheritance_states[paths[0], :]
	else:
		path_states = inheritance_states[paths, :]
		known_indices = np.all(path_states == path_states[0, :], axis=0)
		merged_state[known_indices] = path_states[0, known_indices]
	return merged_state

def viterbi_backward_sweep_autosomes(v_cost, inheritance_states, transition_matrix):

	# backward sweep
	prev_time = time.time()
	n = v_cost.shape[1]
	p, state_len = inheritance_states.p, inheritance_states.state_len
	final_states = -np.ones((state_len, n), dtype=int)
	
	# choose best paths
	# we enforce that the chromosome ends with no deletions and a hard to sequence region
	# also, no de novo deletions
	num_forks = 0
	no_delstates = np.all(inheritance_states[:, [0, 1, 2, 3, -1]]==1, axis=1) & np.all(inheritance_states[:, np.arange(2*inheritance_states.m, 2*inheritance_states.m + 2*(inheritance_states.m - 2))]==0, axis=1)
	min_value = np.min(v_cost[no_delstates, -1])
	paths = np.where(np.isclose(v_cost[:, -1], min_value) & no_delstates)[0]
	print('Num solutions', paths.shape, min_value, inheritance_states[paths, :])

	final_states[:, -1] = merge_paths(paths, inheritance_states)
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
		final_states[:, j] = merge_paths(paths, inheritance_states)
		num_forks += (paths.shape[0] > 1)

	print('Num positions in fork', num_forks)
	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states


