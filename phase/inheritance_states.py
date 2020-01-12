import numpy as np
from itertools import product

# inheritance states
#
# for parents:
# (0, 0) -> deletion on both parental1 and parental2
# (0, 1) -> deletion on parental1
# (1, 0) -> deletion on parental2
# (1, 1) -> normal
# 
# for children:
# (0, 0) -> m1p1
# (0, 1) -> m1p2
# (1, 0) -> m2p1
# (1, 1) -> m2p2
# 
# for de novos:
# (0, 0) -> child has no de novo deletions
# (0, 1) -> child has a paternal de novo deletion
# (1, 0) -> child has a maternal de novo deletion

# flags (last bit)
# 0 -> ok
# 1 -> hard to sequence
# 2 -> low coverage

class AutosomalInheritanceStates:

	def __init__(self, m):
		self.m = m
		states = [x for x in product(*([[0, 1]]*4 + [[0, 1]]*(2*m-4) + [[0, 1]]*(2*m-4) + [[0, 1]]))]

		# child1 inherits (0, 0)
		states = [x for x in states if x[4]==0 and x[5]==0]

		# family can have only a single de novo deletion
		states = [x for x in states if np.sum(x[(2*m):(2*m + 2*(m-2))]) <= 1]

		# can't have a de novo deletion AND an inherited deletion
		states = [x for x in states if (np.sum(x[(2*m):(2*m + 2*(m-2))])==0) or (x[0]==1 and x[1]==1 and x[2]==1 and x[3]==1)]

		## can't have a de novo deletion if we already have an inherited deletion
		#def has_de_novo_and_inh(s):
		#	# find de novo
		#	child_index, is_mat = None, None
		#	for i in range(m-2):
		#		if s[2*m + 2*i] == 1:
		#			child_index = i
		#			is_mat = True
		#		elif s[2*m + 2*i + 1] == 1:
		#			child_index = i
		#			is_pat = True
		#	
		#	# check if this child is already inheriting a deletion
		#	if child_index is not None and s[-1]==2:
		#		return True
		#	elif child_index is not None and is_mat:
		#		return s[s[4 + 2*child_index]]==0
		#	elif child_index is not None and is_pat:
		#		return s[2+s[5 + 2*child_index]]==0
		#	else:
		#		return False
		#states = [x for x in states if not has_de_novo_and_inh(x)]

		# can't have a state where a parent has a deletion that isn't inherited unless it's a double deletion
		states = [x for x in states if x[0]==1 or (x[0]==0 and x[1]==0) or len([i for i in range(4, 2*m, 2) if x[i] == 0])>0]
		states = [x for x in states if x[1]==1 or (x[0]==0 and x[1]==0) or len([i for i in range(4, 2*m, 2) if x[i] == 1])>0]
		states = [x for x in states if x[2]==1 or (x[2]==0 and x[3]==0) or len([i for i in range(5, 2*m, 2) if x[i] == 0])>0]
		states = [x for x in states if x[3]==1 or (x[2]==0 and x[3]==0) or len([i for i in range(5, 2*m, 2) if x[i] == 1])>0]

		self.states = np.asarray(states, dtype=np.int8)
		print('inheritance states', self.states.shape)

		self.p, self.state_len = self.states.shape
		self.state_to_index = dict([(tuple(x), i) for i, x in enumerate(self.states)])

	def __getitem__(self, key):
		return self.states[key]

	def index(self, s):
		return self.state_to_index[s]

	def __contains__(self, key):
		return key in self.state_to_index

