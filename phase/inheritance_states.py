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

	def __init__(self, family, inh_deletions, maternal_phase, paternal_phase, loss_region):
		self._family = family
		assert len(inh_deletions) == 2*family.num_ancestors()
		self._inh_deletions = tuple(inh_deletions)
		assert len(maternal_phase) == len(family)
		self._maternal_phase = tuple(maternal_phase)
		assert len(paternal_phase) == len(family)
		self._paternal_phase = tuple(paternal_phase)
		self._loss_region = loss_region

	@classmethod
	def fromMaternalPhase(self, state, maternal_phase):
		assert len(maternal_phase) == len(state._family)
		return State(state._family, state._inh_deletions, tuple(maternal_phase), state._paternal_phase, state._loss_region)

	@classmethod
	def fromPaternalPhase(self, state, paternal_phase):
		assert len(paternal_phase) == len(state._family)
		return State(state._family, state._inh_deletions, state._maternal_phase, tuple(paternal_phase), state._loss_region)

	@classmethod
	def fromInhDeletions(self, state, inh_deletions):
		assert len(inh_deletions) == 2*state._family.num_ancestors()
		return State(state._family, tuple(inh_deletions), state._maternal_phase, state._paternal_phase, state._loss_region)

	@classmethod
	def fromLossRegion(self, state, loss_region):
		return State(state._family, state._inh_deletions, state._maternal_phase, state._paternal_phase, loss_region)

	def __hash__(self):
		return hash((self._inh_deletions, self._maternal_phase, self._paternal_phase, self._loss_region))

	def __str__(self):
		return str((self._inh_deletions, self._maternal_phase, self._paternal_phase, self._loss_region))

	def __eq__(self, other_state):
		return self._inh_deletions == other_state._inh_deletions and \
			self._maternal_phase == other_state._maternal_phase and \
			self._paternal_phase == other_state._paternal_phase and \
			self._loss_region == other_state._loss_region

	def has_inh_deletion(self, index=None):
		if index is None:
			return np.any([x==0 for x in self._inh_deletions])
		else:
			return self._inh_deletions[index]==0

	def has_inh_duplication(self, index=None):
		if index is None:
			return np.any([x==2 for x in self._inh_deletions])
		else:
			return self._inh_deletions[index]==0

	def has_upd(self, index=None):
		if index is None:
			return np.any([x is not None and x>1 for x in self._maternal_phase + self._paternal_phase])
		else:
			return (self._maternal_phase[index] is not None and self._maternal_phase[index]>1) or (self._paternal_phase[index] is not None and self._paternal_phase[index]>1)

	def has_deletion(self):
		return self.has_inh_deletion()

	def is_hard_to_sequence(self, state):
		return self._loss_region != 0

	def loss_region(self):
		return self._loss_region

	def phase(self):
		phase = np.arange(2*self._family.num_ancestors()).tolist()

		for mom, dad in self._family.ordered_couples:
			mom_index, dad_index = self._family.individual_to_index[mom], self._family.individual_to_index[dad]

			for child in self._family.parents_to_children[(mom, dad)]:
				child_index = self._family.individual_to_index[child]
				# add mat phase
				mat = self._maternal_phase[child_index]
				if mat > 1:
					# UPD
					phase.append(phase[2*dad_index + (mat-2)])
				else:
					phase.append(phase[2*mom_index + mat])
				# add pat phase
				pat = self._paternal_phase[child_index]
				if pat > 1:
					# UPD
					phase.append(phase[2*mom_index + (pat-2)])
				else:
					phase.append(phase[2*dad_index + pat])
		return phase

	def full_state(self):
		full_state = list(self._inh_deletions) + self.phase()
		full_state.append(self._loss_region)
		return full_state

	def __repr__(self):
		return 'State(' +  (' '.join([str(x) for x in self.full_state()])) + ')'


class InheritanceStates:

	def __init__(self, family, detect_mat_inherited_deletions, detect_pat_inherited_deletions, detect_upd, num_loss_states):
		self.family = family

		# 0=deletion, 1=no deletion
		del_options = []
		del_options.extend([[0, 1] if detect_mat_inherited_deletions else [1]]*(2*len(family.mat_ancestors)))
		del_options.extend([[0, 1] if detect_pat_inherited_deletions else [1]]*(2*len(family.pat_ancestors)))
		del_options = list(product(*del_options))
		print('inherited del options', len(del_options))

		mom_to_children, dad_to_children = defaultdict(list), defaultdict(list)
		for (mom, dad), children in self.family.parents_to_children.items():
			mom_to_children[mom].extend(children)
			dad_to_children[dad].extend(children)
		
		# always fix first child of the parent
		parents_with_fixed_child = set()
		#mat_phase_options, pat_phase_options = [[None]]*self.family.num_ancestors(), [[None]]*self.family.num_ancestors()
		phase_options = [[(None, None)]]*self.family.num_ancestors()
		self.fixed_children = []
		for mom, dad in family.ordered_couples:
			for child in family.parents_to_children[(mom, dad)]:
				if mom in parents_with_fixed_child and dad in parents_with_fixed_child:
					if detect_upd:
						phase_options.append([(0, 0), (0, 1), (1, 0), (1, 1),
											  (0, 2), (0, 3), (1, 2), (1, 3),
											  (2, 0), (3, 0), (2, 1), (3, 1),
											  ])
					else:
						phase_options.append([(0, 0), (0, 1), (1, 0), (1, 1)])
				elif mom in parents_with_fixed_child:
					if detect_upd:
						phase_options.append([(0, 0), (1, 0),
											  (0, 2), (0, 3), (1, 2), (1, 3),
											  (2, 0), (3, 0)
											  ])
					else:
						phase_options.append([(0, 0), (1, 0)])
					parents_with_fixed_child.add(dad)
					self.fixed_children.append((child, 'pat'))
				elif dad in parents_with_fixed_child:
					if detect_upd:
						phase_options.append([(0, 0), (0, 1),
											  (0, 2), (0, 3),
											  (2, 0), (3, 0), (2, 1), (3, 1),
											  ])
					else:
						phase_options.append([(0, 0), (0, 1)])
					parents_with_fixed_child.add(mom)
					self.fixed_children.append((child, 'mat'))
				else:
					if detect_upd:
						phase_options.append([(0, 0),
											  (0, 2), (0, 3),
											  (2, 0), (3, 0)
											  ])
					else:
						phase_options.append([(0, 0)])
					parents_with_fixed_child.add(dad)
					self.fixed_children.append((child, 'pat'))
					parents_with_fixed_child.add(mom)
					self.fixed_children.append((child, 'mat'))
					
		phase_options = list(product(*phase_options))
		print('fixed', self.fixed_children)
		print('phase options', len(phase_options))

		loss_options = np.arange(num_loss_states)
		self.num_loss_states = num_loss_states
		print('loss options', len(loss_options))

		self._states = [State(family, inh_deletions,
			[x[0] for x in phase], [x[1] for x in phase], loss_region) for inh_deletions, phase, loss_region in product(del_options, phase_options, loss_options)]

		# you can't have isodisomy if the other parental chromosome has a deletion
		states_to_remove = set()
		for i, state in enumerate(self._states):
			if state.has_inh_deletion() and state.has_upd():
				p = np.array(state.phase())
				for mom, dad in family.ordered_couples:
					mom_index, dad_index = family.individual_to_index[mom], family.individual_to_index[dad]
					#print(state.has_inh_deletion(2*mom_index) or state.has_inh_deletion(2*mom_index+1),
				#		state.has_inh_deletion(2*dad_index) or state.has_inh_deletion(2*dad_index+1),
				#		p[::2], p[1::2])
					if (state.has_inh_deletion(2*mom_index) or state.has_inh_deletion(2*mom_index+1)) and \
						(np.any((p[::2]==2*dad_index) & (p[1::2]==2*dad_index)) or np.any((p[::2]==2*dad_index+1) & (p[1::2]==2*dad_index+1))):
						states_to_remove.add(i)
					if (state.has_inh_deletion(2*dad_index) or state.has_inh_deletion(2*dad_index+1)) and \
						(np.any((p[::2]==2*mom_index) & (p[1::2]==2*mom_index)) or np.any((p[::2]==2*mom_index+1) & (p[1::2]==2*mom_index+1))):
						states_to_remove.add(i)
						
		print('removing isodisomy if other parental chrom has deletion', len(states_to_remove))
		self._states = [x for i, x in enumerate(self._states) if i not in states_to_remove]


		# allow only a single UPD
		self._states = [x for x in self._states if len([y for y in x._maternal_phase+x._paternal_phase if y is not None and y>1])<=1]

		self._states = [x for x in self._states if not (x.has_inh_deletion() and x.has_inh_duplication())]

		self.num_states = len(self._states)
		print('inheritance states', self.num_states)

		self._phase = np.array([x.phase() for x in self._states])
		self._full_states = np.array([x.full_state() for x in self._states])


		self.full_state_length = self._full_states.shape[1]
		self._state_to_index = dict([(x, i) for i, x in enumerate(self._states)])
		self._full_state_to_index = dict([(tuple(x), i) for i, x in enumerate(self._full_states)])

	def index(self, state):
		return self._state_to_index[state]

	def index_from_full_state_tuple(self, state_tuple):
		return self.index(self.get_original_state(state_tuple))

	def __getitem__(self, index):
		return self._states[index]

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
			maternal_phase = np.array(state._maternal_phase)
			ind_indices = self.family.ind_filter(siblings[1:])
			maternal_phase[ind_indices] = [x if x>1 else 1-x for x in maternal_phase[ind_indices]]
			new_state = State.fromMaternalPhase(state, maternal_phase)
			if new_state in self:
				neighbor_indices.add(self.index(new_state))

			# typical maternal recombination in other siblings
			for sibling in siblings[1:]:
				maternal_phase = np.array(state._maternal_phase)
				ind_indices = self.family.ind_filter([sibling])
				maternal_phase[ind_indices] = [x if x>1 else 1-x for x in maternal_phase[ind_indices]]
				new_state = State.fromMaternalPhase(state, maternal_phase)
				if new_state in self:
					neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def get_paternal_recombination_neighbors(self, state):
		neighbor_indices = set()
		for siblings in self.family.parents_to_children.values():
			# first sibling is fixed, so recombination in first sibling results in joint recombination in all other siblings
			paternal_phase = np.array(state._paternal_phase)
			ind_indices = self.family.ind_filter(siblings[1:])
			paternal_phase[ind_indices] = [x if x>1 else 1-x for x in paternal_phase[ind_indices]]
			new_state = State.fromPaternalPhase(state, paternal_phase)
			if new_state != state and new_state in self:
				neighbor_indices.add(self.index(new_state))

			# typical maternal recombination in other siblings
			for sibling in siblings[1:]:
				paternal_phase = np.array(state._paternal_phase)
				ind_indices = self.family.ind_filter([sibling])
				paternal_phase[ind_indices] = [x if x>1 else 1-x for x in paternal_phase[ind_indices]]
				new_state = State.fromPaternalPhase(state, paternal_phase)
				if new_state != state and new_state in self:
					neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def get_upd_neighbors(self, state):
		neighbor_indices = set()

		if state.has_upd():
			# if we're already in a UPD state, we can get out
			for index in range(len(state._family)):
				if state._maternal_phase[index] is not None and state._maternal_phase[index]>1:
					maternal_phase = np.array(state._maternal_phase)
					maternal_phase[index] = 0
					new_state = State.fromMaternalPhase(state, maternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
					maternal_phase[index] = 1
					new_state = State.fromMaternalPhase(state, maternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
				if state._paternal_phase[index] is not None and state._paternal_phase[index]>1:
					paternal_phase = np.array(state._paternal_phase)
					paternal_phase[index] = 0
					new_state = State.fromPaternalPhase(state, paternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
					paternal_phase[index] = 1
					new_state = State.fromPaternalPhase(state, paternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
		else:
			# if we're not in a UPD state, we can get into one
			for index in range(len(state._family)):
				if state._maternal_phase[index] is not None:
					maternal_phase = np.array(state._maternal_phase)
					maternal_phase[index] = 2
					new_state = State.fromMaternalPhase(state, maternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
					maternal_phase[index] = 3
					new_state = State.fromMaternalPhase(state, maternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
				if state._paternal_phase[index] is not None:
					paternal_phase = np.array(state._paternal_phase)
					paternal_phase[index] = 2
					new_state = State.fromPaternalPhase(state, paternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
					paternal_phase[index] = 3
					new_state = State.fromPaternalPhase(state, paternal_phase)
					if new_state in self:
						neighbor_indices.add(self.index(new_state))
		

		return neighbor_indices

	def get_inh_deletion_neighbors(self, state):
		neighbor_indices = []

		for deletion_combination in list(product(*([[0, 1, 2]]*(2*self.family.num_ancestors())))):
			num_dels_changed = np.sum([current_del!=new_del for current_del, new_del in zip(state._inh_deletions, deletion_combination)])
			if num_dels_changed==1:
				new_state = State.fromInhDeletions(state, deletion_combination)
				if new_state in self:
					neighbor_indices.append(self.index(new_state))
		neighbor_indices = np.array(neighbor_indices)
		return neighbor_indices


	def get_loss_neighbors(self, state):
		neighbor_indices = set()
		for i in range(self.num_loss_states):
			new_state = State.fromLossRegion(state, i)
			if new_state != state and new_state in self:
				neighbor_indices.add(self.index(new_state))
		return neighbor_indices

	def is_ok_start(self, state):
		return not state.has_deletion() and not state.has_upd()

	def is_ok_end(self, state):
		return not state.has_deletion() and not state.has_upd()

	def get_full_states(self, state_indices):
		return self._full_states[state_indices, :]

	def get_full_state(self, state_index):
		return self._full_states[state_index, :]

	def get_original_state(self, full_state):
		return self[self._full_state_to_index[tuple(full_state)]]

	def get_perfect_matches(self, state):
		phase = self._phase[self.index(state), :]
		allele_combinations = np.array(list(product(*([[2] if state.has_inh_deletion(i) else [0, 1, 3] if state.has_inh_duplication(i) else [0, 1] for i in range(2*self.family.num_ancestors())]))), dtype=int)

		perfect_matches = list()
		for alleles in allele_combinations:
			perfect_matches.append(tuple(alleles_to_gen[(alleles[phase[2*i]], alleles[phase[2*i + 1]])] for i in range(len(self.family))))
		return perfect_matches, allele_combinations

# 0=no variant, 1=variant, 2=deletion, 3=duplication01
# gens 0=0/0, 1=0/1, 2=1/1, 3=0/-, 4=1/-, 5=-/-
alleles_to_gen = {
	(0, 0): 0, (0, 1): 1, (0, 2): 0, (0, 3): 1,
	(1, 0): 1, (1, 1): 2, (1, 2): 2, (1, 3): 1,
	(2, 0): 0, (2, 1): 2, (2, 2): 5, (2, 3): 1,
	(3, 0): 1, (3, 1): 1, (3, 2): 1, (3, 3): 1,
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
