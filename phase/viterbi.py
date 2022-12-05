import numpy as np
import time
import random

def viterbi_forward_sweep(family_genotypes, mult_factor, states, transition_matrix, loss, allow_del_start=False):
		
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	v_cost = np.zeros((states.num_states, n), dtype=float)
	#v_path = np.zeros((states.num_states, n), dtype=np.int8)
	print('v_cost', v_cost.shape, v_cost.nbytes/10**6, 'MB')
	print('transition_matrix', transition_matrix.transitions.shape)

	v_cost[:, 0] = mult_factor[0]*loss(family_genotypes[:, 0])

	# we enforce that the chromosome starts with no deletions
	if allow_del_start:
		ok_start = np.array([True for x in states])
	else:
		ok_start = np.array([states.is_ok_start(x) for x in states])
	v_cost[~ok_start, 0] = np.inf
	print('ok starts', np.sum(ok_start))

	# next steps
	for j in range(1, n): 
		#v_path[:, j] = np.argmin(v_cost[transition_matrix.transitions, j-1] + transition_matrix.costs, axis=1)
		v_cost[:, j] = np.min(v_cost[transition_matrix.transitions, j-1] + transition_matrix.costs, axis=1) + mult_factor[j]*loss(family_genotypes[:, j])

	print('Forward sweep complete', time.time()-prev_time, 'sec') 
	return v_cost#, v_path

def viterbi_forward_sweep_X(family_genotypes, family_snp_positions, mult_factor, states, transition_matrix, transition_matrixX, loss, assembly, allow_del_start=False):

	if assembly=='37':
		par_end, par_start = 2699520, 154931044
	elif assembly=='38':
		par_end, par_start = 2781479, 155701383

	is_par_loss_region = np.array([x._loss_region==0 or x._loss_region==2 for x in states])
		
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	v_cost = np.zeros((states.num_states, n), dtype=float)
	print('v_cost', v_cost.shape, v_cost.nbytes/10**6, 'MB')
	print('transition_matrix', transition_matrix.transitions.shape)

	v_cost[:, 0] = mult_factor[0]*loss(family_genotypes[:, 0])

	# we enforce that the chromosome starts with no deletions
	if allow_del_start:
		ok_start = np.array([True for x in states])
	else:
		ok_start = np.array([states.is_ok_start(x) for x in states])
	v_cost[~ok_start, 0] = np.inf
	print('ok starts', np.sum(ok_start))

	# next steps
	for j in range(1, n): 
		if (family_snp_positions[j, 0]>= par_end) and (family_snp_positions[j, 1]<=par_start):
			# we're in the X chromosome, but not the PAR
			# so no paternal recombination is allowed, and dad has to have a deletion
			v_cost[:, j] = np.min(v_cost[transition_matrixX.transitions, j-1] + transition_matrixX.costs, axis=1) + mult_factor[j]*loss(family_genotypes[:, j])			
			v_cost[~states._dads_have_deletions, j] = np.inf
			v_cost[is_par_loss_region, j] = np.inf
		else:
			v_cost[:, j] = np.min(v_cost[transition_matrix.transitions, j-1] + transition_matrix.costs, axis=1) + mult_factor[j]*loss(family_genotypes[:, j])
			v_cost[~is_par_loss_region, j] = np.inf

	print('Forward sweep complete', time.time()-prev_time, 'sec') 
	return v_cost

def viterbi_forward_sweep_low_memory(family_genotypes, mult_factor, states, transition_matrix, loss, atol=0.1):
	
	# forward sweep
	prev_time = time.time()

	m, n = family_genotypes.shape
	p = transition_matrix.transitions.shape[1]
	v_path = np.zeros((states.num_states, n), dtype=np.int8)
	print('v_path', v_path.shape, v_path.nbytes/10**6, 'MB')

	v_cost = mult_factor[0]*loss(family_genotypes[:, 0])

	# we enforce that the chromosome starts with no deletions
	ok_start = np.array([states.is_ok_start(x) for x in states])
	v_cost[~ok_start] = np.inf


	# next steps
	indices = np.arange(p)
	for j in range(1, n): 
		path_cost = v_cost[transition_matrix.transitions] + transition_matrix.costs + 10*atol*transition_matrix.is_filler
		min_path_cost = np.tile(np.min(path_cost, axis=1), (p, 1)).T
		
		np.random.shuffle(indices)
		v_path[:, j] = indices[np.argmax(np.isclose(path_cost[:, indices], min_path_cost, rtol=0, atol=atol), axis=1)]
		#v_path[:, j] = np.argmax(np.isclose(path_cost, min_path_cost, rtol=0, atol=0.1), axis=1)

		v_cost = path_cost[np.arange(states.num_states), v_path[:, j]] + mult_factor[j]*loss(family_genotypes[:, j])
		#v_cost = v_cost[transition_matrix.transitions[np.arange(states.num_states), v_path[:, j]]] + transition_matrix.costs[np.arange(states.num_states), v_path[:, j]] + mult_factor[j]*loss(family_genotypes[:, j])
	
	print('Forward sweep complete', time.time()-prev_time, 'sec') 
	return v_path, v_cost


def merge_paths(paths, states):
	# combine path states into a single state (unknown values represented with -1)
	path_states = states.get_full_states(paths)
	return ((path_states[0, :]+1)*np.all(np.equal(path_states, path_states[0, :]), axis=0)) - 1

def merge_ancestral_variants(paths, gen, loss):
	return np.average([loss.get_ancestral_variants(state_index, gen) for state_index in paths], axis=0)


def viterbi_backward_sweep(v_cost, family_genotypes, mult_factor, states, transition_matrix, loss, allow_del_end=False):

	# backward sweep
	prev_time = time.time()
	n = v_cost.shape[1]
	final_states = -np.ones((states.full_state_length, n), dtype=np.int8)
	ancestral_variants = np.zeros((loss.num_acs, n))
	cost = np.zeros((n,))
	print('final_states', final_states.shape, final_states.nbytes/10**6, 'MB')

	# choose best paths
	# we enforce that the chromosome ends with no deletions
	num_forks = 0

	if allow_del_end:
		ok_end = np.array([states.is_ok_end(x) for x in states])
	else:
		ok_end = np.array([states.is_ok_end(x) for x in states])

	min_value = np.min(v_cost[ok_end, -1])
	paths = np.where(np.isclose(v_cost[:, -1], min_value, rtol=0, atol=0.01) & ok_end)[0]
	cost[-1] = np.average(loss(family_genotypes[:, -1])[paths])
	print('Num solutions', paths.shape, min_value)

	if paths.shape[0]>1:
		paths = [random.choice(paths)]

	final_states[:, -1] = merge_paths(paths, states)
	ancestral_variants[:, -1] = merge_ancestral_variants(paths, family_genotypes[:, -1], loss)
	num_forks += (len(paths) > 1)

	# now work backwards
	for j in reversed(range(n-1)):
		# traceback
		total_cost = v_cost[transition_matrix.transitions[paths, :], j] + transition_matrix.costs[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = transition_matrix.transitions[k, np.isclose(total_cost[i, :], min_value[i], rtol=0, atol=0.01)]	
			new_paths.update(min_indices.tolist())
			
		paths = list(new_paths)
		final_states[:, j] = merge_paths(paths, states)
		ancestral_variants[:, j] = merge_ancestral_variants(paths, family_genotypes[:, j], loss)
		cost[j] = np.average(loss(family_genotypes[:, j])[paths])
		num_forks += (len(paths) > 1)

	print('Num forks', num_forks)
	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states, cost, ancestral_variants

def viterbi_backward_sweep_X(v_cost, family_genotypes, family_snp_positions, mult_factor, states, transition_matrix, transition_matrixX, loss, assembly, allow_del_end=False):

	if assembly=='37':
		par_end, par_start = 2699520, 154931044
	elif assembly=='38':
		par_end, par_start = 2781479, 155701383


	# backward sweep
	prev_time = time.time()
	n = v_cost.shape[1]
	final_states = -np.ones((states.full_state_length, n), dtype=np.int8)
	ancestral_variants = np.zeros((loss.num_acs, n))
	cost = np.zeros((n,))
	print('final_states', final_states.shape, final_states.nbytes/10**6, 'MB')

	# choose best paths
	# we enforce that the chromosome ends with no deletions
	num_forks = 0

	if allow_del_end:
		ok_end = np.array([states.is_ok_end(x) for x in states])
	else:
		ok_end = np.array([states.is_ok_end(x) for x in states])

	min_value = np.min(v_cost[ok_end, -1])
	paths = np.where(np.isclose(v_cost[:, -1], min_value, rtol=0, atol=0.01) & ok_end)[0]
	cost[-1] = np.average(loss(family_genotypes[:, -1])[paths])
	print('Num solutions', paths.shape, min_value)

	if paths.shape[0]>1:
		paths = [random.choice(paths)]

	final_states[:, -1] = merge_paths(paths, states)
	ancestral_variants[:, -1] = merge_ancestral_variants(paths, family_genotypes[:, -1], loss)
	num_forks += (len(paths) > 1)

	# now work backwards
	for j in reversed(range(n-1)):
		if (family_snp_positions[j, 0]>= par_end) and (family_snp_positions[j, 1]<=par_start):
			# traceback
			total_cost = v_cost[transition_matrixX.transitions[paths, :], j] + transition_matrixX.costs[paths, :]
			min_value = np.min(total_cost, axis=1)
			new_paths = set()
			for i, k in enumerate(paths):
				min_indices = transition_matrixX.transitions[k, np.isclose(total_cost[i, :], min_value[i], rtol=0, atol=0.01)]	
				new_paths.update(min_indices.tolist())
				
			paths = list(new_paths)
			final_states[:, j] = merge_paths(paths, states)
			ancestral_variants[:, j] = merge_ancestral_variants(paths, family_genotypes[:, j], loss)
			cost[j] = np.average(loss(family_genotypes[:, j])[paths])
			num_forks += (len(paths) > 1)
		else:
			# traceback
			total_cost = v_cost[transition_matrix.transitions[paths, :], j] + transition_matrix.costs[paths, :]
			min_value = np.min(total_cost, axis=1)
			new_paths = set()
			for i, k in enumerate(paths):
				min_indices = transition_matrix.transitions[k, np.isclose(total_cost[i, :], min_value[i], rtol=0, atol=0.01)]	
				new_paths.update(min_indices.tolist())
				
			paths = list(new_paths)
			final_states[:, j] = merge_paths(paths, states)
			ancestral_variants[:, j] = merge_ancestral_variants(paths, family_genotypes[:, j], loss)
			cost[j] = np.average(loss(family_genotypes[:, j])[paths])
			num_forks += (len(paths) > 1)

	print('Num forks', num_forks)
	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states, cost, ancestral_variants

def viterbi_backward_sweep_low_memory(v_path, v_cost, states, transition_matrix):
	# backward sweep
	prev_time = time.time()
	n = v_path.shape[1]
	final_states = -np.ones((states.full_state_length, n), dtype=np.int8)
	print('final_states', final_states.shape, final_states.nbytes/10**6, 'MB')

	# choose best paths
	# we enforce that the chromosome ends with no deletions
	ok_end = np.array([states.is_ok_end(x) for x in states])
	min_value = np.min(v_cost[ok_end])
	paths = np.where(np.isclose(v_cost, min_value, rtol=0, atol=0.1) & ok_end)[0]
	print('Num solutions', paths.shape, min_value, states[paths])

	prev_state = random.choice(paths)
	final_states[:, -1] = states.get_full_state(prev_state)

	# now work backwards
	for j in reversed(range(1, n)):
		# traceback
		prev_state = transition_matrix.transitions[prev_state, v_path[prev_state, j]]
		final_states[:, j-1] = states.get_full_state(prev_state)

	print('Backward sweep complete', time.time()-prev_time, 'sec') 
	
	return final_states


