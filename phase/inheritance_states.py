import numpy as np
from itertools import product, combinations
from collections import defaultdict

# inheritance states
#
# for inherited deletions:
# (0, 0) -> deletion on both parental1 and parental2
# (0, 1) -> deletion on parental1
# (1, 0) -> deletion on parental2
# (1, 1) -> normal
# 
# for inheritance states:
# (0, 2) -> m1p1
# (0, 3) -> m1p2
# (1, 2) -> m2p1
# (1, 3) -> m2p2
# mom is (0, 1) -> m1m2
# dad is (2, 3) -> p1p2
# 

# for hard to sequence regions
# 0 -> ok
# 1 -> hard to sequence

class State:

	def __init__(self, data, inheritance_states):
		self._data = tuple(data)
		self._inheritance_states = inheritance_states

	def __hash__(self):
		return hash(self._data)

	def __str__(self):
		return str(self._data)

	def __eq__(self, other_state):
		return self._data == other_state._data

	def __getitem__(self, index):
		return self._data[index]

	def has_deletion(self, index=None):
		if index is None:
			return np.any([self._data[i]==0 for i in self._inheritance_states.deletion_indices])
		else:
			return self._data[self._inheritance_states.deletion_indices[index]]==0

	def is_hard_to_sequence(self, state):
		return self._data[loss_region_index] != 0

	def toggle_maternal_phase(self, individual):
		index = self._inheritance_states.maternal_phase_indices[self._inheritance_states.family.individuals.index(individual)]
		return State(self._data[:index] + (1-self._data[index],) + self._data[(index+1):], self._inheritance_states)

	def toggle_paternal_phase(self, individual):
		index = self._inheritance_states.paternal_phase_indices[self._inheritance_states.family.individuals.index(individual)]
		return State(self._data[:index] + (1-self._data[index],) + self._data[(index+1):], self._inheritance_states)

	def toggle_ancestral_deletions(self, deletion_states):
		updated_data = np.array(self._data)
		num_dels_changed = np.sum(updated_data != np.array(deletion_states))
		updated_data[self._inheritance_states.deletion_indices] = deletion_states
		return State(updated_data, self._inheritance_states), num_dels_changed

	def toggle_loss(self, new_loss):
		return State(self._data[:self._inheritance_states.loss_region_index] + (new_loss,) + self._data[(self._inheritance_states.loss_region_index+1):], self._inheritance_states)

	def loss_region(self):
		return self._data[self._inheritance_states.loss_region_index]


class InheritanceStates:

	def __init__(self, family, detect_deletions_mat, detect_deletions_pat, num_loss_states):
		self.family = family

		del_options = []
		if detect_deletions_mat:
			del_options.extend([[0, 1]]*(2*len(family.mat_ancestors)))
		else:
			del_options.extend([[1]]*(2*len(family.mat_ancestors)))
		if detect_deletions_pat:
			del_options.extend([[0, 1]]*(2*len(family.pat_ancestors)))
		else:
			del_options.extend([[1]]*(2*len(family.pat_ancestors)))

		mom_to_children, dad_to_children = defaultdict(list), defaultdict(list)
		for (mom, dad), children in self.family.parents_to_children.items():
			mom_to_children[mom].extend(children)
			dad_to_children[dad].extend(children)
		
		# if a parent has more than 2 children, no need to fix the first child - otherwise we need to
		parents_with_fixed_child = set()#[p for p, children in mom_to_children.items() if len(children)>2] + [p for p, children in dad_to_children.items() if len(children)>2])
		phase_options = []
		for mom, dad in family.ordered_couples:
			for child in family.parents_to_children[(mom, dad)]:
				if mom in parents_with_fixed_child:
					phase_options.append([0, 1])
				else:
					phase_options.append([0])
					parents_with_fixed_child.add(mom)

				if dad in parents_with_fixed_child:
					phase_options.append([0, 1])
				else:
					phase_options.append([0])
					parents_with_fixed_child.add(dad)

		loss_regions = [list(np.arange(num_loss_states))]

		self.deletion_indices = np.arange(len(del_options))
		self.maternal_phase_indices = [None]*self.family.num_ancestors() + np.arange(len(del_options), len(del_options)+len(phase_options), 2).tolist()
		self.paternal_phase_indices = [None]*self.family.num_ancestors() + np.arange(len(del_options)+1, len(del_options)+len(phase_options), 2).tolist()
		self.loss_region_index = len(del_options)+len(phase_options)
		self.num_loss_states = num_loss_states

		self._states = np.asarray(list(product(*(del_options + phase_options + loss_regions))), dtype=np.int8)
		self.num_states = self._states.shape[0]
		self._phase = self.get_phase()
		self._full_states = np.hstack((self._states[:, self.deletion_indices], 
			self._phase, 
			self._states[:, self.loss_region_index, np.newaxis]))

		print('inheritance states', self._states.shape)

		self.full_state_length = len(del_options) + 2*len(self.family) + 1
		self._state_to_index = dict([(State(x, self), i) for i, x in enumerate(self._states)])
		self._full_state_to_index = dict([(tuple(x), i) for i, x in enumerate(self._full_states)])

	def index(self, state):
		return self._state_to_index[state]

	def __getitem__(self, index):
		return State(self._states[index, :], self)

	def __contains__(self, state):
		return state in self._state_to_index

	def __iter__(self):
		self.ind = 0
		return self

	def __next__(self):
		if self.ind < self.num_states:
			self.ind += 1
			return self[self.ind-1]
		else:
			raise StopIteration

	def get_maternal_recombination_neighbors(self, state):
		neighbor_indices = set()
		for descendent in self.family.descendents:
			new_state = state.toggle_maternal_phase(descendent)
			if new_state in self:
				neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def get_paternal_recombination_neighbors(self, state):
		neighbor_indices = set()
		for descendent in self.family.descendents:
			new_state = state.toggle_paternal_phase(descendent)
			if new_state in self:
				neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def get_deletion_neighbors(self, state):
		neighbor_indices = []
		neighbor_differences = []
		for deletion_combination in list(product(*([[0, 1]]*len(self.deletion_indices)))):
			new_state, num_dels_changed = state.toggle_ancestral_deletions(deletion_combination)
			if new_state != state and new_state in self:
				neighbor_indices.append(self.index(new_state))
				neighbor_differences.append(num_dels_changed)
		return neighbor_indices, neighbor_differences

	def get_loss_neighbors(self, state):
		neighbor_indices = set()
		for i in range(self.num_loss_states):
			new_state = state.toggle_loss(i)
			if new_state != state and new_state in self:
				neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def is_ok_start(self, state):
		return not state.has_deletion()

	def is_ok_end(self, state):
		return (not state.has_deletion())

	def get_phase(self):
		individual_to_index = dict([(x, i) for i, x in enumerate(self.family.individuals)])
		phase = np.tile(np.arange(2*self.family.num_ancestors()), (self.num_states,1))
		for mom, dad in self.family.ordered_couples:
			mom_index, dad_index = individual_to_index[mom], individual_to_index[dad]

			for child in self.family.parents_to_children[(mom, dad)]:
				child_index = individual_to_index[child]
				mat_phase = phase[np.arange(self.num_states), 2*mom_index + self._states[:, self.maternal_phase_indices[child_index]]]
				pat_phase = phase[np.arange(self.num_states), 2*dad_index + self._states[:, self.paternal_phase_indices[child_index]]]
				phase = np.hstack((phase, mat_phase[:, np.newaxis], pat_phase[:, np.newaxis]))
		return phase

	def get_full_state(self, state_index):
		return tuple(self._full_states[state_index, :])

	def get_original_state(self, full_state):
		return self[self._full_state_to_index[tuple(full_state)]]


	def get_perfect_matches(self, state):
		phase = self._phase[self.index(state), :]
		allele_combinations = list(product(*([[2] if state.has_deletion(i) else [0, 1] for i in range(2*self.family.num_ancestors())])))
		
		perfect_matches = list()
		for alleles in allele_combinations:
			perfect_matches.append(tuple(alleles_to_gen[(alleles[phase[2*i]], alleles[phase[2*i + 1]])] for i in range(len(self.family))))
		return perfect_matches, allele_combinations

# 0=no variant, 1=variant, 2=deletion
alleles_to_gen = {
	(2, 2): 5, (2, 0): 3, (2, 1): 4,
	(0, 2): 3, (0, 0): 0, (0, 1): 1,
	(1, 2): 4, (1, 0): 1, (1, 1): 2,
}

