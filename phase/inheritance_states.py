import numpy as np
from itertools import product, combinations

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
# for de novo deletions:
# (1, 1) -> individual has no de novo deletions
# (1, 0) -> individual has a paternal de novo deletion
# (0, 1) -> individual has a maternal de novo deletion

# for hard to sequence regions
# 0 -> ok
# 1 -> hard to sequence

def inherited_deletion_indices(family_size):
	return np.arange(4)

def maternal_phase_indices(family_size):
	return np.arange(4, 4+(2*family_size), 2)

def paternal_phase_indices(family_size):
	return np.arange(5, 4+(2*family_size), 2)

def hard_to_sequence_index(family_size):
	return 4+(2*family_size)

def preferred_phase_options(family_size):
	#         mom       dad       child1      other children
	return [[0], [1], [2], [3], [0], [2]] + [[0, 1], [2, 3]]*(family_size-3)

class State:

	def __init__(self, family_size, data):
		self._data = data
		self._family_size = family_size

	def __hash__(self):
		return hash(tuple(self._data))

	def __str__(self):
		return str(tuple(self._data))

	def __eq__(self, other_state):
		return tuple(self._data) == tuple(other_state._data)

	def transition_cost(self, other_state, weights):
		cost = sum([weights[i] for i, (x, y) in enumerate(zip(self._data, other_state._data)) if x!=y])
		# preferred phase penalty
		pref = preferred_phase_options(self._family_size)
		for i in range(2*self._family_size):
			if (self._data[4+i] != other_state._data[4+i]) and ((self._data[4+i] not in pref[i]) or (other_state._data[4+i] not in pref[i])):
				cost += 1
		return cost

	def has_deletion(self, ancestral_index=None):
		if ancestral_index is None:
			return np.any(self._data[inherited_deletion_indices(self._family_size)]==0)
		else:
			return self._data[inherited_deletion_indices(self._family_size)[ancestral_index]]==0

	def is_hard_to_sequence(self):
		return self._data[hard_to_sequence_index(self._family_size)]==1

	def get_phase(self, individual_index):
		return self._data[maternal_phase_indices(self._family_size)[individual_index]], self._data[paternal_phase_indices(self._family_size)[individual_index]]

	def add_inherited_deletion(self, ancestral_index):
		self._data[inherited_deletion_indices(self._family_size)[ancestral_index]] = 0

	def remove_inherited_deletion(self, ancestral_index):
		self._data[inherited_deletion_indices(self._family_size)[ancestral_index]] = 1

	def remove_all_inherited_deletions(self):
		self._data[inherited_deletion_indices(self._family_size)] = 1

	def set_maternal_phase(self, individual_index, new_phase):
		self._data[maternal_phase_indices(self._family_size)[individual_index]] = new_phase
		
	def set_paternal_phase(self, individual_index, new_phase):
		self._data[paternal_phase_indices(self._family_size)[individual_index]] = new_phase

	def toggle_hard_to_sequence(self):
		self._data[hard_to_sequence_index(self._family_size)] ^= 1

	



class InheritanceStates:

	def __init__(self, family_size, detect_deletions_mat, detect_deletions_pat):
		self.family_size = family_size

		#if allow_upd:
		#	#                mom       dad       child1                  other children
		#	phase_options = [[0], [1], [2], [3], [0, 2, 3], [0, 1, 2]] + [[0, 1, 2, 3], [0, 1, 2, 3]]*(family_size-3)
		#else:
		#                mom       dad       child1      other children
		phase_options = [[0], [1], [2], [3], [0], [2]] + [[0, 1], [2, 3]]*(family_size-3)
		pref = preferred_phase_options(family_size)

		if detect_deletions_mat and detect_deletions_pat:
			#                              inherited deletions  phase           hard-to-sequence region  
			states = [x for x in product(*([[0, 1]]*4 +         phase_options + [[0, 1]]))]
		elif detect_deletions_mat:
			#                              inherited deletions    phase           hard-to-sequence region  
			states = [x for x in product(*([[0, 1]]*2 + [[1]]*2 + phase_options + [[0, 1]]))]
		elif detect_deletions_pat:
			#                              inherited deletions    phase           hard-to-sequence region  
			states = [x for x in product(*([[1]]*2 + [[0, 1]]*2 + phase_options + [[0, 1]]))]
		else:
			#                              inherited deletions  phase           hard-to-sequence region  
			states = [x for x in product(*([[1]]*4 +         phase_options + [[0, 1]]))]
		
		states = np.asarray(states, dtype=np.int8)

		# family can only have a single UPD event happening at any given position
		num_upd = np.zeros((states.shape[0],))
		for individual_index in range(family_size):
			mat_phase, pat_phase = states[:, maternal_phase_indices(family_size)[individual_index]], states[:, paternal_phase_indices(family_size)[individual_index]]
			mat_preferred_phase, pat_preferred_phase = pref[(2*individual_index):(2*individual_index)+2]
			num_upd += ~np.isin(mat_phase, mat_preferred_phase)
			num_upd += ~np.isin(pat_phase, pat_preferred_phase)

		states = states[num_upd<=1, :]

		# family can have at most 3 inherited deletions 
		# (if 4, you'd expect a stretch of ./. for everyone, and these sites won't be included in VCF)
		states = states[np.sum(states[:, inherited_deletion_indices(family_size)]==0, axis=1)<=3, :]

		# a copy can only have a deletion if it either
		# has at least two partners OR
		# has a partner that is also deleted (involved in a double deletion)
		partners = np.zeros((states.shape[0], 4, 4), dtype=bool)
		in_dd = np.zeros((states.shape[0], 4), dtype=bool)
		for individual_index in range(family_size):
			partners[np.arange(states.shape[0]), states[:, maternal_phase_indices(family_size)[individual_index]], states[:, paternal_phase_indices(family_size)[individual_index]]] = True
			partners[np.arange(states.shape[0]), states[:, paternal_phase_indices(family_size)[individual_index]], states[:, maternal_phase_indices(family_size)[individual_index]]] = True
			
			for i, j in combinations(range(4), 2):
				indices = (states[:, i]==0) & (states[:, j]==0)
				in_dd[indices, i] = True
				in_dd[indices, j] = True
		num_partners = np.sum(partners, axis=2)

		#m1_has_deletion = states[:, inherited_deletion_indices(family_size)[0]]==0
		#m2_has_deletion = states[:, inherited_deletion_indices(family_size)[1]]==0
		#p1_has_deletion = states[:, inherited_deletion_indices(family_size)[2]]==0
		#p2_has_deletion = states[:, inherited_deletion_indices(family_size)[3]]==0

		#states = states[(~m1_has_deletion | (num_partners[:, 0]>=2) | in_dd[:, 0]) & \
		#				(~m2_has_deletion | (num_partners[:, 1]>=2) | in_dd[:, 1]) & \
		#				(~p1_has_deletion | (num_partners[:, 2]>=2) | in_dd[:, 2]) & \
		#				(~p2_has_deletion | (num_partners[:, 3]>=2) | in_dd[:, 3]), :]

		self._states = states
		print('inheritance states', self._states.shape)

		self.num_states = self._states.shape[0]
		self._state_to_index = dict([(State(family_size, x), i) for i, x in enumerate(self._states)])

	def index(self, state):
		return self._state_to_index[state]

	def __getitem__(self, index):
		if isinstance(index, int):
			return State(self.family_size, self._states[index, :])
		else:
			return self._states[index, :]

	def __contains__(self, state):
		return state in self._state_to_index

	def __iter__(self):
		self.index = 0
		return self

	def __next__(self):
		if self.index < self.num_states:
			self.index += 1
			return self[self.index-1]
		else:
			raise StopIteration

	def is_ok_start_end(self, state):
		return not state.has_deletion() and state.is_hard_to_sequence()

class YInheritanceStates:

	def __init__(self, family_size, individual_sex):
		self.family_size = family_size

		#                 mom      dad
		phase_options = [[0], [0], [2], [2]]

		is_first_boy = True
		for sex in individual_sex[2:]:
			if sex=='1' and is_first_boy:
				# if child is a boy
				phase_options.extend([[2], [2]])
				is_first_boy = False
			elif sex=='1':
				phase_options.extend([[2], [2]])
			else:
				phase_options.extend([[0], [0]])

		#                              inherited deletions    phase           hard-to-sequence region  
		states = [x for x in product(*([[0], [0], [1], [1]] + phase_options + [[0, 1]]))]
		states = np.asarray(states, dtype=np.int8)

		self._states = states
		print('inheritance states', self._states.shape)

		self.num_states = self._states.shape[0]
		self._state_to_index = dict([(State(family_size, x), i) for i, x in enumerate(self._states)])

	def index(self, state):
		return self._state_to_index[state]

	def __getitem__(self, index):
		if isinstance(index, int):
			return State(self.family_size, self._states[index, :])
		else:
			return self._states[index, :]

	def __contains__(self, state):
		return state in self._state_to_index

	def __iter__(self):
		self.index = 0
		return self

	def __next__(self):
		if self.index < self.num_states:
			self.index += 1
			return self[self.index-1]
		else:
			raise StopIteration

	def is_ok_start_end(self, state):
		return state.is_hard_to_sequence()




