import numpy as np
from itertools import product
#from collections import Counter

# genotype (pred, obs): cost
g_cost = {
	(-1, -1): 0,
	(-1, 0): 1,
	(-1, 1): 1,
	(-1, 2): 1,
	(0, -1): 0,
	(0, 0): 0,
	(0, 1): 1,
	(0, 2): 2,
	(1, -1): 0,
	(1, 0): 1,
	(1, 1): 0,
	(1, 2): 1,
	(2, -1): 0,
	(2, 0): 2,
	(2, 1): 1,
	(2, 2): 0
}

class LazyLoss:
	def __init__(self, m, inheritance_states, genotypes, cached=True):

		self.__build_perfect_matches__(m, inheritance_states)
		self.__build_loss_equivalence__(inheritance_states)

		self.genotypes = genotypes
		self.cached = cached
		p, q = inheritance_states.p, genotypes.q

		if cached:
			self.losses = np.zeros((int(p/4) if m>=5 else p, q), dtype=np.int8)
			self.already_calculated = np.zeros((q,), dtype=bool)
		else:
			self.l = np.zeros((int(p/4) if m>=5 else p,))
		self.s = np.zeros((self.num_pm_gen,), dtype=np.int8)
		
	def __call__(self, gen): 
		gen_index = self.genotypes.index(gen)
		if self.cached:

			if not self.already_calculated[gen_index]:
				for i, pm in enumerate(self.pm_gen):
					self.s[i] = sum([g_cost[(pred, obs)] for pred, obs in zip(pm, gen)])
				    
				for i, indices in enumerate(self.pm_gen_indices):
					self.losses[i, gen_index] = np.min(self.s[indices])
				self.already_calculated[gen_index] = True
			return self.losses[self.full_loss_indices, gen_index].astype(int)

		else:
			for i, pm in enumerate(self.pm_gen):
				self.s[i] = sum([g_cost[(pred, obs)] for pred, obs in zip(pm, gen)])
				    
			for i, indices in enumerate(self.pm_gen_indices):
				self.l[i] = np.min(self.s[indices])
			return self.l[self.full_loss_indices].astype(int)

	def __build_perfect_matches__(self, m, inheritance_states):
		# perfect match genotypes
		pm_gen_to_index = dict()
		self.pm_gen_indices = []
		for s in inheritance_states:
			if s[4] == 0 and s[5] == 0:
				anc_pos = [[-1] if s[i] == 0 else [0, 1] for i in range(4)]
				anc_variants = np.array(list(product(*anc_pos)), dtype=np.int8)
				pred_gens = np.zeros((anc_variants.shape[0], m), dtype=np.int8)

				# mom
				# deletion
				pred_gens[(anc_variants[:, 0]==-1) & (anc_variants[:, 1]==-1), 0] = -1
				pred_gens[(anc_variants[:, 0]==-1) & (anc_variants[:, 1]==0), 0] = 0
				pred_gens[(anc_variants[:, 0]==-1) & (anc_variants[:, 1]==1), 0] = 2
				pred_gens[(anc_variants[:, 0]==0) & (anc_variants[:, 1]==-1), 0] = 0
				pred_gens[(anc_variants[:, 0]==1) & (anc_variants[:, 1]==-1), 0] = 2
				# normal
				pred_gens[(anc_variants[:, 0]==0) & (anc_variants[:, 1]==0), 0] = 0
				pred_gens[(anc_variants[:, 0]==1) & (anc_variants[:, 1]==1), 0] = 2
				pred_gens[(anc_variants[:, 0]==0) & (anc_variants[:, 1]==1), 0] = 1
				pred_gens[(anc_variants[:, 0]==1) & (anc_variants[:, 1]==0), 0] = 1

				# dad
				# deletion
				pred_gens[(anc_variants[:, 2]==-1) & (anc_variants[:, 3]==-1), 1] = -1
				pred_gens[(anc_variants[:, 2]==-1) & (anc_variants[:, 3]==0), 1] = 0
				pred_gens[(anc_variants[:, 2]==-1) & (anc_variants[:, 3]==1), 1] = 2
				pred_gens[(anc_variants[:, 2]==0) & (anc_variants[:, 3]==-1), 1] = 0
				pred_gens[(anc_variants[:, 2]==1) & (anc_variants[:, 3]==-1), 1] = 2
				# normal
				pred_gens[(anc_variants[:, 2]==0) & (anc_variants[:, 3]==0), 1] = 0
				pred_gens[(anc_variants[:, 2]==1) & (anc_variants[:, 3]==1), 1] = 2
				pred_gens[(anc_variants[:, 2]==0) & (anc_variants[:, 3]==1), 1] = 1
				pred_gens[(anc_variants[:, 2]==1) & (anc_variants[:, 3]==0), 1] = 1

				# children
				for index in range(m-2):
					mat, pat = s[(4+(2*index)):(6+(2*index))]

					# deletion
					pred_gens[(anc_variants[:, mat]==-1) & (anc_variants[:, 2+pat]==-1), 2+index] = -1
					pred_gens[(anc_variants[:, mat]==-1) & (anc_variants[:, 2+pat]==0), 2+index] = 0
					pred_gens[(anc_variants[:, mat]==-1) & (anc_variants[:, 2+pat]==1), 2+index] = 2
					pred_gens[(anc_variants[:, mat]==0) & (anc_variants[:, 2+pat]==-1), 2+index] = 0
					pred_gens[(anc_variants[:, mat]==1) & (anc_variants[:, 2+pat]==-1), 2+index] = 2
					# normal
					pred_gens[(anc_variants[:, mat]==0) & (anc_variants[:, 2+pat]==0), 2+index] = 0
					pred_gens[(anc_variants[:, mat]==1) & (anc_variants[:, 2+pat]==1), 2+index] = 2
					pred_gens[(anc_variants[:, mat]==0) & (anc_variants[:, 2+pat]==1), 2+index] = 1
					pred_gens[(anc_variants[:, mat]==1) & (anc_variants[:, 2+pat]==0), 2+index] = 1

				unique_pred_gens = set(map(tuple, pred_gens))
				for pg in unique_pred_gens:
					if pg not in pm_gen_to_index:
						pm_gen_to_index[pg] = len(pm_gen_to_index)
				self.pm_gen_indices.append([pm_gen_to_index[pg] for pg in unique_pred_gens])

		self.num_pm_gen = len(pm_gen_to_index)
		self.pm_gen = np.zeros((self.num_pm_gen, m), dtype=np.int8)
		for pm, i in pm_gen_to_index.items():
			self.pm_gen[i, :] = pm

		print('perfect matches', self.pm_gen.shape)#, Counter([len(v) for v in self.pm_gen_indices]))

	def __build_loss_equivalence__(self, inheritance_states):
		# losses are symmetrical to parental chromosome swaps
		self.full_loss_indices = np.zeros((inheritance_states.p,), dtype=int)
		loss_state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states[(inheritance_states[:, 4]==0) & (inheritance_states[:, 5]==0), :])])
		for i, s in enumerate(inheritance_states):
			new_s = -np.ones((inheritance_states.state_len,), dtype=np.int8)
			if s[4] == 0:
				new_s[:2] = s[:2]
				new_s[np.arange(4, inheritance_states.state_len, 2)] = s[np.arange(4, inheritance_states.state_len, 2)]
			else:
				new_s[:2] = s[[1, 0]]
				new_s[np.arange(4, inheritance_states.state_len, 2)] = 1-s[np.arange(4, inheritance_states.state_len, 2)]
			if s[5] == 0:
				new_s[2:4] = s[2:4]
				new_s[np.arange(5, inheritance_states.state_len, 2)] = s[np.arange(5, inheritance_states.state_len, 2)]
			else:
				new_s[2:4] = s[[3, 2]]
				new_s[np.arange(5, inheritance_states.state_len, 2)] = 1-s[np.arange(5, inheritance_states.state_len, 2)]
			self.full_loss_indices[i] = loss_state_to_index[tuple(new_s)]