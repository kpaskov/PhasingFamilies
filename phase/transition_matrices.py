import numpy as np
from itertools import product

class TransitionMatrix:
	# This class represents a transition matrix from every state to every other.
	# We limit state transitions, so for efficiency we store transitions as a matrix 
	# where row i lists all of the neighboring states of state i. The transition_costs
	# matrix stores the corresponding transition costs.

	def __init__(self, states, params):

		# transition matrix
		transitions = [[] for i in range(states.num_states)]
		transition_costs = [[] for i in range(states.num_states)]
		for state_index, state in enumerate(states):

			# allow a transition into the same state for free
			transitions[state_index].append(state_index)
			transition_costs[state_index].append(0)

			## allow a transition into or out of one or more haplotypes
			#neighbors = states.get_haplotype_neighbors(state)
			#transitions[state_index].extend(neighbors)
			#transition_costs[state_index].extend([params['-log10(P[haplotype_entry_exit])']]*len(neighbors))
			#num_hap_neighbors = len(neighbors)

			# allow a transition into or out of one or more inherited deletions
			neighbors = states.get_inh_deletion_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[inherited_deletion_entry_exit])']]*len(neighbors))
			num_del_neighbors = len(neighbors)

			# allow a transition into or out of one or more denovo deletions
			neighbors = states.get_denovo_deletion_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[denovo_deletion_entry_exit])']]*len(neighbors))
			
			# allow a transition into or out of one or more inherited duplications
			#neighbors, num_changed = states.get_duplication_neighbors(state)
			#transitions[state_index].extend(neighbors)
			#transition_costs[state_index].extend([params['-log10(P[inherited_deletion_entry_exit])']*n for n in num_changed])
			#num_dup_neighbors = len(neighbors)
			
			# allow a single recombination event
			neighbors = states.get_maternal_recombination_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[maternal_crossover])']]*len(neighbors))
			num_recomb_neighbors = len(neighbors)

			neighbors = states.get_paternal_recombination_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[paternal_crossover])']]*len(neighbors))
			num_recomb_neighbors += len(neighbors)

			# allow a single UPD transition
			neighbors = states.get_upd_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([20]*len(neighbors))

			# allow a transition into or out of a new loss region
			neighbors = states.get_loss_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[loss_transition])']]*len(neighbors))
			num_loss_neighbors = len(neighbors)

			#print(num_hap_neighbors, num_del_neighbors, num_dup_neighbors, num_recomb_neighbors, num_loss_neighbors)

			# update cost of not transitioning (it will be close to 0, but not exactly)
			p_transition = np.sum([10**(-x) for x in transition_costs[state_index][1:]])
			transition_costs[state_index][0] = -np.log10(1-p_transition)
		            
		# transitions is a ragged matrix - square it off
		max_trans = max([len(x) for x in transitions])
		self.is_filler = np.zeros((len(transitions), max_trans))
		for i, (t, c) in enumerate(zip(transitions, transition_costs)):
			while len(t) < max_trans:
				self.is_filler[i, len(t)] = True
				t.append(t[-1])
				c.append(c[-1])

		self.transitions = np.array(transitions)
		self.costs = np.array(transition_costs)

		if np.sum(self.costs<0)>0:
			print('Negative transmission costs!')
			self.costs[self.costs<0] = 0
		#assert np.all(self.costs>0)

		
		print('transitions', self.transitions.shape)

class TransitionMatrixX:
	# This class represents a transition matrix from every state to every other.
	# We limit state transitions, so for efficiency we store transitions as a matrix 
	# where row i lists all of the neighboring states of state i. The transition_costs
	# matrix stores the corresponding transition costs.

	def __init__(self, states, params):

		# transition matrix
		transitions = [[] for i in range(states.num_states)]
		transition_costs = [[] for i in range(states.num_states)]
		for state_index, state in enumerate(states):

			# allow a transition into the same state for free
			transitions[state_index].append(state_index)
			transition_costs[state_index].append(0)

			# allow a transition into or out of one or more inherited deletions
			neighbors = states.get_inh_deletion_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[inherited_deletion_entry_exit])']]*len(neighbors))
			num_del_neighbors = len(neighbors)

			# allow a transition into or out of one or more denovo deletions
			neighbors = states.get_denovo_deletion_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[denovo_deletion_entry_exit])']]*len(neighbors))
			
			# allow a single recombination event
			neighbors = states.get_maternal_recombination_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[maternal_crossover])']]*len(neighbors))
			num_recomb_neighbors = len(neighbors)

			# paternal recombination is not allowed in the PAR
			#neighbors = states.get_paternal_recombination_neighbors(state)
			#transitions[state_index].extend(neighbors)
			#transition_costs[state_index].extend([params['-log10(P[paternal_crossover])']]*len(neighbors))
			#num_recomb_neighbors += len(neighbors)

			# allow a single UPD transition
			neighbors = states.get_upd_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([20]*len(neighbors))

			# allow a transition into or out of a new loss region
			neighbors = states.get_loss_neighbors(state)
			transitions[state_index].extend(neighbors)
			transition_costs[state_index].extend([params['-log10(P[loss_transition])']]*len(neighbors))
			num_loss_neighbors = len(neighbors)

			#print(num_hap_neighbors, num_del_neighbors, num_dup_neighbors, num_recomb_neighbors, num_loss_neighbors)

			# update cost of not transitioning (it will be close to 0, but not exactly)
			p_transition = np.sum([10**(-x) for x in transition_costs[state_index][1:]])
			transition_costs[state_index][0] = -np.log10(1-p_transition)
		            
		# transitions is a ragged matrix - square it off
		max_trans = max([len(x) for x in transitions])
		self.is_filler = np.zeros((len(transitions), max_trans))
		for i, (t, c) in enumerate(zip(transitions, transition_costs)):
			while len(t) < max_trans:
				self.is_filler[i, len(t)] = True
				t.append(t[-1])
				c.append(c[-1])

		self.transitions = np.array(transitions)
		self.costs = np.array(transition_costs)

		if np.sum(self.costs<0)>0:
			print('Negative transmission costs!')
			self.costs[self.costs<0] = 0
		#assert np.all(self.costs>0)

		
		print('transitions', self.transitions.shape)



