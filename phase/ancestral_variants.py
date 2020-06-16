import numpy as np
from itertools import product
from collections import Counter

class AncestralVariants:
	def __init__(self, states, genotypes, loss, family, state):
		# given a particular state for a family, this class lets us calculate the most likely set of underlying
		# ancestral variants for each genotype

		#  0=no variant, 1=variant, 2=deletion
		perfect_matches, allele_combinations = states.get_perfect_matches(state)
		self.perfect_matches = np.array(perfect_matches, dtype=np.int8)
		self.allele_combinations = np.array(allele_combinations, dtype=np.int8)
		self.best_match = -np.ones((2*family.num_ancestors(), len(genotypes)), dtype=np.int8)
		self.already_calculated = np.zeros((len(genotypes),), dtype=bool)
		self._loss = loss
		self._state = state
		self._genotypes = genotypes
		self.family_size = len(family)

	def merge_alleles(self, allele_indices):
		# combine alleles into a single state (unknown values represented with -1)
		merged_allele = -np.ones((self.best_match.shape[0],), dtype=np.int8)
		alleles = self.allele_combinations[allele_indices, :]
		known_indices = np.all(alleles == alleles[0, :], axis=0)
		merged_allele[known_indices] = alleles[0, known_indices]
		return merged_allele

	def __call__(self, gen): 
		gen_index = self._genotypes.index(gen)
		if not self.already_calculated[gen_index]:
			perfect_match_costs = np.zeros((len(self.perfect_matches),))
			for i, pm in enumerate(self.perfect_matches):
				perfect_match_costs[i] = np.sum(self._loss.emission_params[self._state.loss_region(), np.arange(self.family_size), list(pm), list(gen)])
			best_allele_indices = np.where(np.isclose(perfect_match_costs, np.min(perfect_match_costs)))[0]
			self.best_match[:, gen_index] = self.merge_alleles(best_allele_indices)
			self.already_calculated[gen_index] = True
		return self.best_match[:, gen_index]

