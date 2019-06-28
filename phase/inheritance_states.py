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

# flags (last bit)
# 0 -> ok
# 1 -> hard to sequence
# 2 -> low coverage

class AutosomalInheritanceStates:

	def __init__(self, m):
		self.m = m
		states = [x for x in product(*([[0, 1]]*4 + [[0, 1]]*(2*m-4) + [[0, 1, 2]]))]

		# child1 inherits (0, 0)
		states = [x for x in states if x[4]==0 and x[5]==0]

		# can't have a state where a parent has a deletion that isn't inherited or a double deletion
		states = [x for x in states if x[0]==1 or x[0]==2 or (x[0]==0 and x[1]==0) or len([i for i in range(4, 2*m, 2) if x[i] == 0])>0]
		states = [x for x in states if x[1]==1 or x[1]==2 or (x[0]==0 and x[1]==0) or len([i for i in range(4, 2*m, 2) if x[i] == 1])>0]
		states = [x for x in states if x[2]==1 or x[2]==2 or (x[2]==0 and x[3]==0) or len([i for i in range(5, 2*m, 2) if x[i] == 0])>0]
		states = [x for x in states if x[3]==1 or x[3]==2 or (x[2]==0 and x[3]==0) or len([i for i in range(5, 2*m, 2) if x[i] == 1])>0]

		# can't have a state with a deletion in a low-coverage region
		states = [x for x in states if x[-1] != 2 or np.sum(x[:4])>=3]
		print(len(states))
		
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

