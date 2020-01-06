import numpy as np
from itertools import product

class Genotypes:
	def __init__(self, m):
		self.genotypes = np.array([x for x in list(product(*[[-1, 0, 1, 2]]*m)) if np.any(np.array(x)>0) or np.all(np.array(x)==0)], dtype=np.int8)
		self.genotype_to_index = dict([(tuple(x), i) for i, x in enumerate(self.genotypes)])
		self.q = self.genotypes.shape[0]
		print('genotypes', self.genotypes.shape)

	def __getitem__(self, key):
		return self.genotypes[key]

	def index(self, s):
		return self.genotype_to_index[s]

	def __contains__(self, key):
		return key in self.genotype_to_index