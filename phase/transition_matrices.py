import numpy as np
from itertools import product

class AutosomalTransitionMatrix:

	def __init__(self, inheritance_states, params):

		self.shift_costs = 	[params['-log10(P[deletion_entry_exit])']]*4 + \
			[params['-log10(P[maternal_crossover])'] if i%2==0 else params['-log10(P[paternal_crossover])'] for i in range(2*(inheritance_states.m-2))] + \
			[params['-log10(P[hard_to_seq_region_entry_exit])']]

		p, state_len = inheritance_states.p, inheritance_states.state_len

		# starting costs
		starting_state = (1, 1, 1, 1, 0, 0)
		self.first_costs = np.zeros((p,), dtype=float)
		for i, ss in enumerate(starting_state):
			self.first_costs[inheritance_states[:, i] != ss] += self.shift_costs[i]

		# transition matrix
		transitions = [[] for i in range(p)]
		transition_costs = [[] for i in range(p)]
		for i, state in enumerate(inheritance_states):
			for delstate in list(product(*[[0, 1]]*4)):
				new_state = tuple(delstate) + tuple(state[4:])
				if new_state in inheritance_states:
					new_index = inheritance_states.index(new_state)
					transitions[i].append(new_index)
					transition_costs[i].append(sum([self.shift_costs[j] for j, (old_s, new_s) in enumerate(zip(state[:4], delstate)) if old_s != new_s]))

			# allow a single recombination event (if we're not in a deletion)
			if state[0]==1 and state[1]==1 and state[2]==1 and state[3]==1:
				for j in range(4, inheritance_states.state_len-1):
					new_state = tuple(1-x if k == j else x for k, x in enumerate(state))
					if new_state in inheritance_states:
						new_index = inheritance_states.index(new_state)
						transitions[i].append(new_index)
						transition_costs[i].append(self.shift_costs[j])

			# allow a transition into hard_to_sequence regions (as long as we're not in a deletion)
			if np.all(state[:4]==1):
				for o, n in [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)]:
					if state[-1] == o:
						new_state = tuple(n if k == inheritance_states.state_len-1 else x for k, x in enumerate(state))

						if new_state in inheritance_states:
							new_index = inheritance_states.index(new_state)
							transitions[i].append(new_index)
							transition_costs[i].append(self.shift_costs[-1])
		            
		# transitions is a ragged matrix - square it off
		max_trans = max([len(x) for x in transitions])
		for t, c in zip(transitions, transition_costs):
			while len(t) < max_trans:
				t.append(t[-1])
				c.append(c[-1])

		self.transitions = np.array(transitions)
		self.costs = np.array(transition_costs)

		
		print('transitions', self.transitions.shape)

class XTransitionMatrix:

	def __init__(self, inheritance_states, shift_costs, sex_of_children):

		# We'll need different transition matrices depending on where we are on the X-chrom

		self.shift_costs = shift_costs
		p, state_len = inheritance_states.p, inheritance_states.state_len

		# starting costs
		starting_state = [1, 1, 1, 1, 0]
		if sex_of_children[0] == '2':
			# first child is female
			starting_state.append(0)
		else:
			starting_state.append(1)
		self.first_costs = np.zeros((p,), dtype=int)
		for i, ss in enumerate(starting_state):
			self.first_costs[inheritance_states[:, i] != ss] += shift_costs[i]

		########################### in-PAR transition matrix ###########################
		transitions = [[] for i in range(inheritance_states.p)]
		transition_costs = [[] for i in range(inheritance_states.p)]
		for i, state in enumerate(inheritance_states):
			for delstate in list(product(*[[0, 1]]*4)):
				new_state = tuple(delstate) + tuple(state[4:])
				if new_state in inheritance_states:
					new_index = inheritance_states.index(new_state)
					transitions[i].append(new_index)
					transition_costs[i].append(sum([shift_costs[j] for j, (old_s, new_s) in enumerate(zip(state[:4], delstate)) if old_s != new_s]))

			# allow a single recombination event
			for j in range(4, inheritance_states.state_len):
				new_state = tuple(1-x if k == j else x for k, x in enumerate(state))
				if new_state in inheritance_states:
					new_index = inheritance_states.index(new_state)
					transitions[i].append(new_index)
					transition_costs[i].append(shift_costs[j])
		            
		self.transitions_inPAR = np.array(transitions)
		self.costs_inPAR = np.array(transition_costs)

		########################### out-PAR transition matrix ###########################

		# first, decide which states are allowed outside the PAR
		self.out_par_states = []
		for i, s in enumerate(inheritance_states):
			# females inherit p1, males inherit p2
			inheritance_ok = True
			for j, sex_of_child in enumerate(sex_of_children):
				if sex_of_child == '2':
					# female child
					inheritance_ok = inheritance_ok and (s[(2*j)+5] == 0)
				if sex_of_child == '1':
					# male child
					inheritance_ok = inheritance_ok and (s[(2*j)+5] == 1)
		            
			# we enforce a deletion on p2
			if inheritance_ok and s[3]==0:
				self.out_par_states.append(i)

		r = len(self.out_par_states)
		self.r = r
		out_par_state_to_index = dict([(i, x) for x, i in enumerate(self.out_par_states)])
		#print('states allowed outside PAR', r, '\nstates allowed inside PAR', p)

		# now restrict transition matrix to those states
		out_par_transitions, out_par_transition_costs = [], []
		for i in range(p):
			if i in self.out_par_states:
				trans, costs = [], []
				for t, c in zip(self.transitions_inPAR[i, :], self.costs_inPAR[i, :]):
					if t in self.out_par_states:
						trans.append(out_par_state_to_index[t])
						costs.append(c)
		                
				out_par_transitions.append(trans)
				out_par_transition_costs.append(costs)
		self.transitions_outPAR = np.array(out_par_transitions)
		self.costs_outPAR = np.array(out_par_transition_costs)

		########################### to-PAR and from-PAR transition matrix ###########################
		self.transitions_fromPAR = np.zeros((r, p), dtype=int)
		self.transitions_toPAR = np.zeros((p, r), dtype=int)
		self.costs_fromtoPAR = np.zeros((r, p), dtype=int)

		for i in range(p):
			self.transitions_fromPAR[:, i] = i
		    
		for i in range(r):
			self.transitions_toPAR[:, i] = i
		    
		for i in range(p):
			for j in range(r):
				self.costs_fromtoPAR[j, i] = np.abs(inheritance_states[i, :]-inheritance_states[self.out_par_states[j], :]).dot(shift_costs)

		#print('in-PAR transitions', self.transitions_inPAR.shape, 'out-PAR transitions', self.transitions_outPAR.shape)
		#print('from-PAR transitions', self.transitions_fromPAR.shape, 'to-PAR transitions', self.transitions_toPAR.shape)

