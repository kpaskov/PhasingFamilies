import numpy as np
from itertools import product

class Genotype:

	def __init__(self, data):
		self._data = tuple(data)

	def __hash__(self):
		return hash(self._data)

	def __str__(self):
		return str(self._data)

	def __eq__(self, other_state):
		return self._data == other_state._data

	def __getitem__(self, index):
		return self._data[index]

	def af_region(self):
		return self._data[-1]

class Genotypes:
	def __init__(self, m, af_boundaries):
		self._genotypes = np.array([x for x in list(product(*([[-1, 0, 1, 2]]*m + [list(range(len(af_boundaries)))]))) if np.any(np.array(x[:-1])>0) or np.all(np.array(x)==0)], dtype=np.int8)
		self._genotype_to_index = dict([(Genotype(x), i) for i, x in enumerate(self._genotypes)])
		self.num_genotypes = self._genotypes.shape[0]
		print('genotypes', self._genotypes.shape)

	def index(self, gen):
		return self._genotype_to_index[Genotype(gen)]

	def __contains__(self, gen):
		return Genotype(gen) in self._genotype_to_index

	def __len__(self):
		return self.num_genotypes
