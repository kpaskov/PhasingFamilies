import numpy as np
from itertools import product, combinations, chain
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

	def has_duplication(self, index=None):
		if index is None:
			return np.any([self._data[i]==2 for i in self._inheritance_states.deletion_indices])
		else:
			return self._data[self._inheritance_states.deletion_indices[index]]==2

	def has_haplotype(self, index=None):
		if index is None:
			return np.any([self._data[i]>=3 for i in self._inheritance_states.deletion_indices])
		else:
			return self._data[self._inheritance_states.deletion_indices[index]]>=3

	def haplotype_off(self):
		updated_data = np.array(self._data)
		indices = [i for i in self._inheritance_states.deletion_indices if self._data[i]>=3]
		updated_data[indices] = 1
		return State(updated_data, self._inheritance_states)

	def get_haplotype(self, index):
		return None if not self.has_haplotype(index) else self._data[self._inheritance_states.deletion_indices[index]]-3

	def is_hard_to_sequence(self, state):
		return self._data[loss_region_index] != 0

	def toggle_maternal_phases(self, individuals):
		updated_data = np.array(self._data)
		indices = [self._inheritance_states.maternal_phase_indices[self._inheritance_states.family.individuals.index(individual)] for individual in individuals]
		updated_data[indices] = 1-updated_data[indices]
		return State(updated_data, self._inheritance_states)

	def toggle_paternal_phases(self, individuals):
		updated_data = np.array(self._data)
		indices = [self._inheritance_states.paternal_phase_indices[self._inheritance_states.family.individuals.index(individual)] for individual in individuals]
		updated_data[indices] = 1-updated_data[indices]
		return State(updated_data, self._inheritance_states)

	def toggle_ancestral_deletions(self, deletion_states):
		updated_data = np.array(self._data)
		num_dels_changed = np.sum(updated_data[self._inheritance_states.deletion_indices] != np.array(deletion_states))
		updated_data[self._inheritance_states.deletion_indices] = deletion_states
		return State(updated_data, self._inheritance_states), num_dels_changed

	def toggle_loss(self, new_loss):
		return State(self._data[:self._inheritance_states.loss_region_index] + (new_loss,) + self._data[(self._inheritance_states.loss_region_index+1):], self._inheritance_states)

	def loss_region(self):
		return self._data[self._inheritance_states.loss_region_index]


class InheritanceStates:

	def __init__(self, family, detect_deletions_mat, detect_deletions_pat, 
								detect_duplications_mat, detect_duplications_pat, 
								num_loss_states):
		self.family = family

		s = np.arange(2*family.num_ancestors())
		haplotype_combinations = list(chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1)))

		# 0=deletion, 2=duplication, 3+=haplotype
		del_options = []
		if detect_deletions_mat and detect_duplications_mat:
			base_options = [0, 1, 2]
		elif detect_deletions_mat:
			base_options = [0, 1]
		elif detect_duplications_mat:
			base_options = [0, 2]
		else:
			base_options = [1]
		#del_options.extend([base_options]*(2*len(family.mat_ancestors)))
		del_options.extend([base_options + (3+np.arange(i)).tolist() for i in np.arange(2*len(family.mat_ancestors))])

		if detect_deletions_pat and detect_duplications_pat:
			base_options = [0, 1, 2]
		elif detect_deletions_pat:
			base_options = [0, 1]
		elif detect_duplications_pat:
			base_options = [0, 2]
		else:
			base_options = [1]
		#del_options.extend([base_options]*(2*len(family.pat_ancestors)))
		del_options.extend([base_options + (3+np.arange(2*len(family.mat_ancestors)+i)).tolist() for i in np.arange(2*len(family.pat_ancestors))])

		print(del_options)

		mom_to_children, dad_to_children = defaultdict(list), defaultdict(list)
		for (mom, dad), children in self.family.parents_to_children.items():
			mom_to_children[mom].extend(children)
			dad_to_children[dad].extend(children)
		
		# always fix first child of the parent
		parents_with_fixed_child = set()
		phase_options = []
		self.fixed_children = []
		for mom, dad in family.ordered_couples:
			for child in family.parents_to_children[(mom, dad)]:
				if mom in parents_with_fixed_child:
					phase_options.append([0, 1])
				else:
					phase_options.append([0])
					parents_with_fixed_child.add(mom)
					self.fixed_children.append((child, 'mat'))

				if dad in parents_with_fixed_child:
					phase_options.append([0, 1])
				else:
					phase_options.append([0])
					parents_with_fixed_child.add(dad)
					self.fixed_children.append((child, 'pat'))

		loss_regions = [list(np.arange(num_loss_states))]
		print('fixed', self.fixed_children)

		self.deletion_indices = np.arange(len(del_options))
		self.maternal_phase_indices = [None]*self.family.num_ancestors() + np.arange(len(del_options), len(del_options)+len(phase_options), 2).tolist()
		self.paternal_phase_indices = [None]*self.family.num_ancestors() + np.arange(len(del_options)+1, len(del_options)+len(phase_options), 2).tolist()
		self.loss_region_index = len(del_options)+len(phase_options)
		self.num_loss_states = num_loss_states

		self._states = np.asarray(list(product(*(del_options + phase_options + loss_regions))), dtype=np.int8)

		# can only have one deletion or duplication in the family at any position
		# self._states = self._states[np.sum(np.isin(self._states[:, self.deletion_indices], [0, 2]), axis=1) <= 1, :]

		# can't have a deletion/duplication at the same position
		#self._states = self._states[np.all(self._states[:, self.deletion_indices]<3, axis=1) | np.all(~np.isin(self._states[:, self.deletion_indices], [0, 2]), axis=1), :]

		# must chain multiple haplotypes - can't have two pointing to the same place
		for i in np.arange(2*family.num_ancestors()):
			self._states = self._states[np.sum(self._states[:, self.deletion_indices]==3+i, axis=1)<=1, :]

		# deletion/duplication must be inherited
		self.num_states = self._states.shape[0]
		phase = self.get_phase()
		print(self._states.shape, phase.shape)
		is_inherited = np.ones((self.num_states,), dtype=bool)
		for i in self.deletion_indices:
			is_inherited[((self._states[:, i]==0) | (self._states[:, i]==2)) & (np.sum(phase[:, 2*self.family.num_ancestors():]==i, axis=1)==0)] = False
		self._states = self._states[is_inherited, :]

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

	def index_from_full_state_tuple(self, state_tuple):
		return self.index(self.get_original_state(state_tuple))

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
		for siblings in self.family.parents_to_children.values():
			# first sibling is fixed, so recombination in first sibling results in joint recombination in all other siblings
			new_state = state.toggle_maternal_phases(siblings[1:])
			if new_state in self:
				neighbor_indices.add(self.index(new_state))

			# typical maternal recombination in other siblings
			for sibling in siblings[1:]:
				new_state = state.toggle_maternal_phases([sibling])
				if new_state in self:
					neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def get_paternal_recombination_neighbors(self, state):
		neighbor_indices = set()
		for siblings in self.family.parents_to_children.values():
			# first sibling is fixed, so recombination in first sibling results in joint recombination in all other siblings
			new_state = state.toggle_paternal_phases(siblings[1:])
			if new_state in self:
				neighbor_indices.add(self.index(new_state))

			# typical paternal recombination in other siblings
			for sibling in siblings[1:]:
				new_state = state.toggle_paternal_phases([sibling])
				if new_state in self:
					neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def get_deletion_neighbors(self, state):
		neighbor_indices = []
		neighbor_differences = []

		for deletion_combination in list(product(*([[0, 1] if state[i] == 1 or state[i] == 0 else [state[i]] for i in self.deletion_indices]))):
			new_state, num_dels_changed = state.toggle_ancestral_deletions(deletion_combination)
			if new_state != state and new_state in self:
				neighbor_indices.append(self.index(new_state))
				neighbor_differences.append(num_dels_changed)
		neighbor_indices = np.array(neighbor_indices)
		neighbor_differences = np.array(neighbor_differences)
		return neighbor_indices[neighbor_differences==1], neighbor_differences[neighbor_differences==1]

	def get_duplication_neighbors(self, state):
		neighbor_indices = []
		neighbor_differences = []

		for deletion_combination in list(product(*([[2, 1] if state[i] == 1 or state[i] == 2 else [state[i]] for i in self.deletion_indices]))):
			new_state, num_dels_changed = state.toggle_ancestral_deletions(deletion_combination)
			if new_state != state and new_state in self:
				neighbor_indices.append(self.index(new_state))
				neighbor_differences.append(num_dels_changed)
		neighbor_indices = np.array(neighbor_indices)
		neighbor_differences = np.array(neighbor_differences)
		return neighbor_indices[neighbor_differences==1], neighbor_differences[neighbor_differences==1]

	def get_haplotype_neighbors(self, state):
		neighbor_indices = []
		if state.has_haplotype():
			new_state = state.haplotype_off()
			if new_state != state and new_state in self:
				neighbor_indices.append(self.index(new_state))
		else:
			for hap_combination in list(product(*([[1] + (3+np.arange(len(self.deletion_indices))).tolist() if state[i]==1 else [state[i]] for i in self.deletion_indices]))):
				new_state, num_dels_changed = state.toggle_ancestral_deletions(hap_combination)
				if new_state != state and new_state in self:
					neighbor_indices.append(self.index(new_state))
		neighbor_indices = np.array(neighbor_indices)
		return neighbor_indices

	def get_loss_neighbors(self, state):
		neighbor_indices = set()
		for i in range(self.num_loss_states):
			new_state = state.toggle_loss(i)
			if new_state != state and new_state in self:
				neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def is_ok_start(self, state):
		return (not state.has_deletion()) and (not state.has_duplication())

	def is_ok_end(self, state):
		return (not state.has_deletion()) and (not state.has_duplication())

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

	def get_full_states(self, state_indices):
		return self._full_states[state_indices, :]

	def get_original_state(self, full_state):
		return self[self._full_state_to_index[tuple(full_state)]]

	def get_perfect_matches(self, state):
		phase = self._phase[self.index(state), :]
		allele_combinations = np.array(list(product(*([[2] if state.has_deletion(i) else [3, 4, 5] if state.has_duplication(i) else [-1] if state.has_haplotype(i) else [0, 1] for i in range(2*self.family.num_ancestors())]))), dtype=int)

		# replace haplotypes
		allele_combinations_haplotype_replaced = allele_combinations.copy()
		for i in np.arange(2*self.family.num_ancestors()):
			if state.has_haplotype(i):
				indices = allele_combinations_haplotype_replaced[:, i]==-1
				allele_combinations_haplotype_replaced[indices, i] = allele_combinations_haplotype_replaced[indices, state.get_haplotype(i)] 

		perfect_matches = list()
		for alleles in allele_combinations_haplotype_replaced:
			perfect_matches.append(tuple(alleles_to_gen[(alleles[phase[2*i]], alleles[phase[2*i + 1]])] for i in range(len(self.family))))
		return perfect_matches, allele_combinations

# 0=no variant, 1=variant, 2=deletion, 3=duplication00, 4=duplication01, 5=duplication11
# gens 0=0/0, 1=0/1, 2=1/1, 3=0/-, 4=1/-, 5=-/-, 6=0/
alleles_to_gen = {
	(0, 0): 0, (0, 1): 1, (0, 2): 0, (0, 3): 0, (0, 4): 1, (0, 5): 1,
	(1, 0): 1, (1, 1): 2, (1, 2): 2, (1, 3): 1, (1, 4): 1, (1, 5): 2,
	(2, 0): 0, (2, 1): 2, (2, 2): 5, (2, 3): 0, (2, 4): 1, (2, 5): 2,
	(3, 0): 0, (3, 1): 1, (3, 2): 0, (3, 3): 0, (3, 4): 1, (3, 5): 1,
	(4, 0): 1, (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1, (4, 5): 1,
	(5, 0): 1, (5, 1): 2, (5, 2): 2, (5, 3): 1, (5, 4): 1, (5, 5): 2
}

# 0, 1, 3
# 00 -> 0
# 01 -> 1
# 03 -> 1
# 10 -> 1
# 11 -> 2
# 13 -> 1
# 30 -> 1
# 31 -> 1
# 33 -> 1
