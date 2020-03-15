import numpy as np
from itertools import product

class AutosomalTransitionMatrix:

	def __init__(self, inheritance_states, params):

		self.shift_costs = 	[params['-log10(P[inherited_deletion_entry_exit])']]*4 + \
			[params['-log10(P[maternal_crossover])'] if i%2==0 else params['-log10(P[paternal_crossover])'] for i in range(2*(inheritance_states.m-2))] + \
			[params['-log10(P[denovo_deletion_entry_exit])'] for i in range(2*(inheritance_states.m-2))]
		self.hard_to_sequence_cost = params['-log10(P[hard_to_seq_region_entry_exit])']
		self.low_coverage_cost = params['-log10(P[low_coverage_region_entry_exit])']

		print(len(self.shift_costs))
		p = inheritance_states.p

		# transition matrix
		transitions = [[] for i in range(p)]
		transition_costs = [[] for i in range(p)]
		for i, state in enumerate(inheritance_states):
			# allow a transition into the same state
			transitions[i].append(i)
			transition_costs[i].append(0)

			# allow a transition into or out of one or more deletions
			for delstate in list(product(*[[0, 1] for x in state[:4]])):
				new_state = tuple(delstate) + tuple(state[4:])
				if tuple(state) != new_state and new_state in inheritance_states:
					new_index = inheritance_states.index(new_state)
					transitions[i].append(new_index)
					transition_costs[i].append(np.sum([self.shift_costs[j] for j, (old_s, new_s) in enumerate(zip(state[:4], delstate)) if old_s != new_s]))

			# allow a single recombination event (if we're not in a deletion, including de novo)
			for j in range(4, 2*inheritance_states.m):
				if (j%2==0 and np.all(state[:2]==1) and np.all(state[np.arange(2*inheritance_states.m, 2*inheritance_states.m + 2*(inheritance_states.m -2), 2)]==0)) or (j%2==1 and np.all(state[2:4]==1) and np.all(state[np.arange(2*inheritance_states.m+1, 2*inheritance_states.m + 2*(inheritance_states.m -2), 2)]==0)):
					new_state = tuple(1-x if k == j else x for k, x in enumerate(state))
					if new_state in inheritance_states:
						new_index = inheritance_states.index(new_state)
						transitions[i].append(new_index)
						transition_costs[i].append(self.shift_costs[j])

			# allow a single de novo event
			for j in range(2*inheritance_states.m, 2*inheritance_states.m + 2*(inheritance_states.m-2)):
				new_state = tuple(1-x if k == j else x for k, x in enumerate(state))
				if new_state in inheritance_states:
					new_index = inheritance_states.index(new_state)
					transitions[i].append(new_index)
					transition_costs[i].append(self.shift_costs[j])

			# allow a transition into or out of a hard_to_sequence region
			for o, n in [(0, 1), (1, 0)]:
				if state[-1] == o:
					new_state = tuple(state[:-1]) + (n,)

					if new_state in inheritance_states:
						new_index = inheritance_states.index(new_state)
						transitions[i].append(new_index)
						transition_costs[i].append(self.hard_to_sequence_cost)

			# allow a transition into or out of a low coverage region
			for o, n in [(0, 2), (2, 0)]:
				if state[-1] == o:
					new_state = tuple(state[:-1]) + (n,)

					if new_state in inheritance_states:
						new_index = inheritance_states.index(new_state)
						transitions[i].append(new_index)
						transition_costs[i].append(self.low_coverage_cost)

			# update cost of not transitioning (it will be close to 0, but not exactly)
			p_transition = np.sum([10**(-x) for x in transition_costs[i][1:]])
			transition_costs[i][0] = -np.log10(1-p_transition)
		            
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


