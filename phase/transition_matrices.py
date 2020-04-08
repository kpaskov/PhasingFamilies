import numpy as np
from itertools import product
import copy

class TransitionMatrix:
	# This class represents a transition matrix from every state to every other.
	# We limit state transitions, so for efficiency we store transitions as a matrix 
	# where row i lists all of the neighboring states of state i. The transition_costs
	# matrix stores the corresponding transition costs.

	def __init__(self, states, params):

		shift_costs = [params['-log10(P[inherited_deletion_entry_exit])']]*4 + \
			[params['-log10(P[maternal_crossover])'] if i%2==0 else params['-log10(P[paternal_crossover])'] for i in range(2*(states.family_size))] + \
		    [params['-log10(P[hard_to_seq_region_entry_exit])']]

		print(shift_costs)

		# transition matrix
		transitions = [[] for i in range(states.num_states)]
		transition_costs = [[] for i in range(states.num_states)]
		for state_index in range(states.num_states):
			state = states[state_index]

			# allow a transition into the same state for free
			transitions[state_index].append(state_index)
			transition_costs[state_index].append(0)

			# allow a transition into or out of one or more inherited deletions
			for deletion_combination in list(product(*([[True, False]]*4))):
				new_state = copy.deepcopy(state)
				for i, has_del in enumerate(deletion_combination):
					if has_del:
						new_state.add_inherited_deletion(i)
					else:
						new_state.remove_inherited_deletion(i)

				if state != new_state and new_state in states:
					transitions[state_index].append(states.index(new_state))
					transition_costs[state_index].append(state.transition_cost(new_state, shift_costs))

			# allow a single recombination event
			for individual_index in range(states.family_size):
				for i in range(4):
					# maternal
					new_state = copy.deepcopy(state)
					new_state.remove_all_inherited_deletions() # can't have recombination inside a deletion, but we allow a transition into no deletions+recombination
					new_state.set_maternal_phase(individual_index, i)
					if state != new_state and new_state in states:
						transitions[state_index].append(states.index(new_state))
						transition_costs[state_index].append(state.transition_cost(new_state, shift_costs))

					# paternal
					new_state = copy.deepcopy(state)
					new_state.remove_all_inherited_deletions() # can't have recombination inside a deletion, but we allow a transition into no deletions+recombination
					new_state.set_paternal_phase(individual_index, i)
					if state != new_state and new_state in states:
						transitions[state_index].append(states.index(new_state))
						transition_costs[state_index].append(state.transition_cost(new_state, shift_costs))

			# allow a transition into or out of a hard_to_sequence region
			new_state = copy.deepcopy(state)
			new_state.toggle_hard_to_sequence()
			if state != new_state and new_state in states:
				transitions[state_index].append(states.index(new_state))
				transition_costs[state_index].append(state.transition_cost(new_state, shift_costs))


			# update cost of not transitioning (it will be close to 0, but not exactly)
			p_transition = np.sum([10**(-x) for x in transition_costs[state_index][1:]])
			transition_costs[state_index][0] = -np.log10(1-p_transition)
		            
		# transitions is a ragged matrix - square it off
		max_trans = max([len(x) for x in transitions])
		for t, c in zip(transitions, transition_costs):
			while len(t) < max_trans:
				t.append(t[-1])
				c.append(c[-1])

		self.transitions = np.array(transitions)
		self.costs = np.array(transition_costs)
		assert np.all(self.costs>0)

		
		print('transitions', self.transitions.shape)


