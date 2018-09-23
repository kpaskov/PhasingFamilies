import numpy as np
import time

PAR1X_end = 2699520
PAR2X_start = 154931044

def viterbi_forward_sweep_autosomes(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, loss):
		
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	p, state_len = inheritance_states.p, inheritance_states.state_len
	v_cost = np.zeros((p, n), dtype=int)

	# first step, break symmetry
	# we enforce that the chromosome starts with child1 (0, 0) and no deletions or duplications
	pos_gen = tuple(family_genotypes[:, 0])
	v_cost[:, 0] = mult_factor[0]*loss(pos_gen) + transition_matrix.first_costs

	# next steps
	for j in range(1, n): 
		pos_gen = tuple(family_genotypes[:, j])
		v_cost[:, j] = np.min(v_cost[transition_matrix.transitions, j-1] + transition_matrix.costs, axis=1) + mult_factor[j]*loss(pos_gen)

	print('Forward sweep complete', time.time()-prev_time, 'sec') 

	return v_cost

def merge_paths(paths, inheritance_states):
	# combine path states a single state (unknown values represented with -1)
	merged_state = np.zeros((inheritance_states.state_len,), dtype=np.int8)
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
	# we enforce that the chromosome ends with no deletions
	num_forks = 0
	no_delstates = np.all(inheritance_states[:, :4]==1, axis=1)
	min_value = np.min(v_cost[no_delstates, -1])
	paths = np.where((v_cost[:, -1]==min_value) & no_delstates)[0]
	print('Num solutions', paths.shape, inheritance_states[paths, :])

	final_states[:, -1] = merge_paths(paths, inheritance_states)
	num_forks += (paths.shape[0] > 1)

	# now work backwards
	for j in reversed(range(n-1)):
		# traceback
		total_cost = v_cost[transition_matrix.transitions[paths, :], j] + transition_matrix.costs[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = transition_matrix.transitions[k, np.where(total_cost[i, :] == min_value[i])[0]]	
			new_paths.update(min_indices.tolist())
		
		paths = np.asarray(list(new_paths), dtype=int)
		final_states[:, j] = merge_paths(paths, inheritance_states)
		num_forks += (paths.shape[0] > 1)

	print('Num positions in fork', num_forks)
	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states

def viterbi_forward_sweep_X(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, loss):
		
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	p, state_len = inheritance_states.p, inheritance_states.state_len
	r = transition_matrix.r

	last_in_par1 = np.argmax(family_snp_positions[:, 0] > PAR1X_end)
	first_in_par2 = np.argmax(family_snp_positions[:, 0] > PAR2X_start)
	
	PAR1_v_cost = np.zeros((p, last_in_par1), dtype=int)
	outPAR_v_cost = np.zeros((r, first_in_par2-last_in_par1), dtype=int)
	PAR2_v_cost = np.zeros((p, n-first_in_par2), dtype=int)

	# first step, break symmetry
	# we enforce that the chromosome starts with child1 (0, ) and no deletions or duplications
	pos_gen = tuple(family_genotypes[:, 0])
	PAR1_v_cost[:, 0] = mult_factor[0]*loss(pos_gen) + transition_matrix.first_costs

	# PAR1
	for j in range(1, last_in_par1): 
		pos_gen = tuple(family_genotypes[:, j])
		PAR1_v_cost[:, j] = np.min(PAR1_v_cost[transition_matrix.transitions_inPAR, j-1] + transition_matrix.costs_inPAR, axis=1) + mult_factor[j]*loss(pos_gen)

	# Transition out of PAR
	pos_gen = tuple(family_genotypes[:, last_in_par1])
	outPAR_v_cost[:, 0] = np.min(PAR1_v_cost[transition_matrix.transitions_fromPAR, -1] + transition_matrix.costs_fromtoPAR, axis=1) + mult_factor[last_in_par1]*loss(pos_gen)[transition_matrix.out_par_states]

	# Out of PAR
	for j in range(last_in_par1+1, first_in_par2): 
		pos_gen = tuple(family_genotypes[:, j])
		outPAR_v_cost[:, j-last_in_par1] = np.min(outPAR_v_cost[transition_matrix.transitions_outPAR, j-1-last_in_par1] + transition_matrix.costs_outPAR, axis=1) + mult_factor[j]*loss(pos_gen)[transition_matrix.out_par_states]

	# Transition into PAR
	pos_gen = tuple(family_genotypes[:, first_in_par2])
	PAR2_v_cost[:, 0] = np.min(outPAR_v_cost[transition_matrix.transitions_toPAR, -1] + transition_matrix.costs_fromtoPAR.T, axis=1) + mult_factor[first_in_par2]*loss(pos_gen)

	# PAR2
	for j in range(first_in_par2+1, n): 
		pos_gen = tuple(family_genotypes[:, j])
		PAR2_v_cost[:, j-first_in_par2] = np.min(PAR2_v_cost[transition_matrix.transitions_inPAR, j-1-first_in_par2] + transition_matrix.costs_inPAR, axis=1) + mult_factor[j]*loss(pos_gen)

	print('Forward sweep complete', time.time()-prev_time, 'sec') 

	return PAR1_v_cost, outPAR_v_cost, PAR2_v_cost, last_in_par1, first_in_par2

def viterbi_backward_sweep_X(PAR1_v_cost, outPAR_v_cost, PAR2_v_cost, last_in_par1, first_in_par2, inheritance_states, transition_matrix):

	# backward sweep
	prev_time = time.time()
	n = PAR1_v_cost.shape[1] + outPAR_v_cost.shape[1] + PAR2_v_cost.shape[1]
	p, state_len = inheritance_states.p, inheritance_states.state_len
	final_states = -np.ones((state_len, n), dtype=int)
	out_par_state_to_index = dict([(i, x) for x, i in enumerate(transition_matrix.out_par_states)])
	
	# first step, choose best paths
	# we enforce that the chromosome ends with no deletions
	num_forks = 0
	no_delstates = np.all(inheritance_states[:, :4]==1, axis=1)
	min_value = np.min(PAR2_v_cost[no_delstates, -1])
	paths = np.where((PAR2_v_cost[:, -1]==min_value) & no_delstates)[0]
	print('Num solutions', paths.shape, inheritance_states[paths, :])

	final_states[:, -1] = merge_paths(paths, inheritance_states)
	num_forks += (paths.shape[0] > 1)

	# PAR2
	for j in reversed(range(PAR2_v_cost.shape[1]-1)):

		# traceback
		total_cost = PAR2_v_cost[transition_matrix.transitions_inPAR[paths, :], j] + transition_matrix.costs_inPAR[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = transition_matrix.transitions_inPAR[k, np.where(total_cost[i, :] == min_value[i])[0]]
			new_paths.update(min_indices.tolist())
		
		paths = np.asarray(list(new_paths), dtype=int)
		final_states[:, j+first_in_par2] = merge_paths(paths, inheritance_states)
		num_forks += (paths.shape[0] > 1)
	
	# Transition out of PAR
	total_cost = outPAR_v_cost[transition_matrix.transitions_toPAR[paths, :], 0] + transition_matrix.costs_fromtoPAR.T[paths, :]
	min_value = np.min(total_cost, axis=1)
	new_paths = set()
	for i, k in enumerate(paths):
		min_indices = transition_matrix.transitions_toPAR[k, np.where(total_cost[i, :] == min_value[i])[0]]
		new_paths.update([transition_matrix.out_par_states[i] for i in min_indices.tolist()])
	
	paths = np.asarray(list(new_paths), dtype=int)
	final_states[:, first_in_par2-1] = merge_paths(paths, inheritance_states)
	num_forks += (paths.shape[0] > 1)

	# Out of PAR
	for j in reversed(range(outPAR_v_cost.shape[1]-1)):

		# traceback
		paths = [out_par_state_to_index[p] for p in paths]
		total_cost = outPAR_v_cost[transition_matrix.transitions_outPAR[paths, :], j] + transition_matrix.costs_outPAR[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = transition_matrix.transitions_outPAR[k, np.where(total_cost[i, :] == min_value[i])[0]]
			new_paths.update([transition_matrix.out_par_states[i] for i in min_indices.tolist()])
		
		paths = np.asarray(list(new_paths), dtype=int)
		final_states[:, j+last_in_par1] = merge_paths(paths, inheritance_states)
		num_forks += (paths.shape[0] > 1)
	
	# Transition into PAR
	paths = [out_par_state_to_index[p] for p in paths]
	total_cost = PAR1_v_cost[transition_matrix.transitions_fromPAR[paths, :], 0] + transition_matrix.costs_fromtoPAR[paths, :]
	min_value = np.min(total_cost, axis=1)
	new_paths = set()
	for i, k in enumerate(paths):
		min_indices = transition_matrix.transitions_fromPAR[k, np.where(total_cost[i, :] == min_value[i])[0]]
		new_paths.update(min_indices.tolist())

	paths = np.asarray(list(new_paths), dtype=int)
	final_states[:, last_in_par1-1] = merge_paths(paths, inheritance_states)
	num_forks += (paths.shape[0] > 1)
	    
	# PAR1
	for j in reversed(range(PAR1_v_cost.shape[1]-1)):

		total_cost = PAR1_v_cost[transition_matrix.transitions_inPAR[paths, :], j] + transition_matrix.costs_inPAR[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = transition_matrix.transitions_inPAR[k, np.where(total_cost[i, :] == min_value[i])[0]]
			new_paths.update(min_indices.tolist())
		
		paths = np.asarray(list(new_paths), dtype=int)
		final_states[:, j] = merge_paths(paths, inheritance_states)
		num_forks += (paths.shape[0] > 1)

	print('Num positions in fork', num_forks)
	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states


