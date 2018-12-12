import numpy as np
from itertools import product

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

# This code takes advantage of the fact that our loss is symmetric in the sense that the state (1, 1, 1, 1, 0, 1, 0, 1)
# will have equivalent losses for all genotypes as (1, 1, 1, 1, 1, 0, 1, 0). Essentially, we can swap parental
# chromosomes without a problem.

# We arbitrarily fix positions 4 and 5 (the chromosomal inheritance of child 1) to be 0, 0 when calculating our
# perfect matches and then we cache a mapping between equivalent states with the __build_loss_equivalence__ function.
# This mapping allows us to produce losses for all possible states after only explicitly calculating the losses
# for one quarter of them. This trick reduces both compute and memory by a factor of 1/4. Since this is our most
# memory intensive piece of code, this is very useful.

class LazyLoss:
	def __init__(self, m, inheritance_states, genotypes):

		self.m = m
		self.p, self.q, self.state_len = inheritance_states.p, genotypes.q, inheritance_states.state_len
		self.genotypes = genotypes
		self.inheritance_states = inheritance_states
		
		self.pm_gen_indices = -np.ones((int(self.p/4) if m>=5 else self.p, 16), dtype=int)
		self.parental_variants = dict() # (state, gen) -> parental variants
		self.__build_perfect_matches__(m)

		self.__build_loss_equivalence__()
		self.losses = np.zeros((int(self.p/4) if m>=5 else self.p, self.q), dtype=np.int8)

		print('losses', self.losses.shape)
		self.already_calculated = np.zeros((self.q,), dtype=bool)
		
		self.s = np.zeros((self.num_pm_gen,), dtype=np.int8)
		
	def __call__(self, gen): 
		gen_index = self.genotypes.index(gen)
		if not self.already_calculated[gen_index]:
			for i, pm in enumerate(self.pm_gen):
				self.s[i] = sum([g_cost[(pred, obs)] for pred, obs in zip(pm, gen)])
				    
			self.losses[:, gen_index] = np.min(self.s[self.pm_gen_indices], axis=1)
			self.already_calculated[gen_index] = True
		return self.losses[self.full_loss_indices, gen_index].astype(int)

	def get_parental_variants(self, state, gen):
		if (state, gen) in self.parental_variants:
			# these are perfect matches - huge majority of cases we encounter
			return self.parental_variants[(state, gen)], 0, tuple((0,)*self.m)
		else:
			# these are imperfect, we calculate on the fly
			gen_index = self.genotypes.index(gen)
			new_state_index = self.full_loss_indices[self.inheritance_states.index(state)]
			s = self.loss_states[new_state_index]

			# will we need to flip mom or dad?
			flip_mom = (state[4] == 1)
			flip_dad = (state[5] == 1)

			min_value, min_indices, min_blame = None, [], []
			for i, pm in enumerate(self.pm_gen[self.pm_gen_indices[new_state_index, :]]):
				blame = [g_cost[(pred, obs)] for pred, obs in zip(pm, gen)]
				v = sum(blame)
				if min_value is None or v < min_value:
					min_value = v
					min_indices = [i]
					min_blame = [blame]
				elif v == min_value:
					min_indices.append(i)
					min_blame.append(blame)

			# calculate parental variants
			anc_pos = [[-1, -1] if s[i] == 0 else [0, 1] for i in range(4)]
			all_options = np.array(list(product(*anc_pos)), dtype=np.int8)[min_indices, :]
			pv = -np.ones((4,), dtype=np.int8)
			pv[s[:4]==0] = -2
			pv[np.all(all_options==0, axis=0)] = 0
			pv[np.all(all_options==1, axis=0)] = 1

			if flip_mom:
				pv = pv[[1, 0, 2, 3]]
			if flip_dad:
				pv = pv[[0, 1, 3, 2]]

			# distribute blame
			if min_value == 0:
				blame = (0.0,)*self.m
			else:
				blame = np.sum(np.asarray(min_blame), axis=0)
				blame = min_value*blame/np.sum(blame) # normalize

			return tuple(pv), int(min_value), tuple(blame)


	def __build_perfect_matches__(self, m):
		# perfect match genotypes
		pm_gen_to_index = dict()

		self.loss_states = self.inheritance_states[(self.inheritance_states[:, 4]==0) & (self.inheritance_states[:, 5]==0), :]
		for i, s in enumerate(self.loss_states):
			if s[4] == 0 and s[5] == 0:
				anc_pos = [[-1, -1] if s[i] == 0 else [0, 1] for i in range(4)]
				anc_variants = np.array(list(product(*anc_pos)), dtype=np.int8)
				pred_gens = np.zeros((16, m), dtype=np.int8)

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

				# transform pred_gens (perfect matches) to indices
				for j, pg in enumerate(map(tuple, pred_gens)):
					if pg not in pm_gen_to_index:
						pm_gen_to_index[pg] = len(pm_gen_to_index)
					self.pm_gen_indices[i, j] = pm_gen_to_index[pg]

				unique_pgs, indices = np.unique(pred_gens, axis=0, return_inverse=True)
				for j, pg in enumerate(unique_pgs):
					all_options = anc_variants[np.where(indices==j)[0], :]
					pv = -np.ones((4,), dtype=np.int8)
					pv[s[:4]==0] = -2
					pv[np.all(all_options==0, axis=0)] = 0
					pv[np.all(all_options==1, axis=0)] = 1

					self.parental_variants[(tuple(s), tuple(pg))] = tuple(pv)

		# Now flip parental variants to include swapped parental chromosomes
		new_parental_variants = dict()
		for (s, gen), pv in self.parental_variants.items():
			new_s = -np.ones((self.state_len,), dtype=np.int8)

			# flip mom, dad stays the same
			new_s[:4] = [s[i] for i in [1, 0, 2, 3]]
			new_pv = tuple(pv[i] for i in [1, 0, 2, 3])
			new_s[np.arange(4, self.state_len, 2)] = [1-s[i] for i in range(4, self.state_len, 2)]
			new_s[np.arange(5, self.state_len, 2)] = [s[i] for i in range(5, self.state_len, 2)]
			new_parental_variants[(tuple(new_s), gen)] = new_pv

			# flip dad, mom stays the same
			new_s[:4] = [s[i] for i in [0, 1, 3, 2]]
			new_pv = tuple(pv[i] for i in [0, 1, 3, 2])
			new_s[np.arange(4, self.state_len, 2)] = [s[i] for i in range(4, self.state_len, 2)]
			new_s[np.arange(5, self.state_len, 2)] = [1-s[i] for i in range(5, self.state_len, 2)]
			new_parental_variants[(tuple(new_s), gen)] = new_pv

			# flip mom and dad
			new_s[:4] = [s[i] for i in [1, 0, 3, 2]]
			new_pv = tuple(pv[i] for i in [1, 0, 3, 2])
			new_s[np.arange(4, self.state_len, 2)] = [1-s[i] for i in range(4, self.state_len, 2)]
			new_s[np.arange(5, self.state_len, 2)] = [1-s[i] for i in range(5, self.state_len, 2)]
			new_parental_variants[(tuple(new_s), gen)] = new_pv

		self.parental_variants.update(new_parental_variants)

		self.num_pm_gen = len(pm_gen_to_index)
		self.pm_gen = np.zeros((self.num_pm_gen, m), dtype=np.int8)
		for pm, i in pm_gen_to_index.items():
			self.pm_gen[i, :] = pm

		print('perfect matches', self.pm_gen.shape)#, Counter([len(v) for v in self.pm_gen_indices]))
		print('parental_variants', len(self.parental_variants))

	def __build_loss_equivalence__(self):
		# losses are symmetrical to parental chromosome swaps
		self.full_loss_indices = np.zeros((self.p,), dtype=int)
		loss_state_to_index = dict([(tuple(x), i) for i, x in enumerate(self.loss_states)])
		for i, s in enumerate(self.inheritance_states):
			new_s = -np.ones((self.state_len,), dtype=np.int8)
			if s[4] == 0:
				new_s[:2] = s[:2]
				new_s[np.arange(4, self.state_len, 2)] = s[np.arange(4, self.state_len, 2)]
			else:
				new_s[:2] = s[[1, 0]]
				new_s[np.arange(4, self.state_len, 2)] = 1-s[np.arange(4, self.state_len, 2)]
			if s[5] == 0:
				new_s[2:4] = s[2:4]
				new_s[np.arange(5, self.state_len, 2)] = s[np.arange(5, self.state_len, 2)]
			else:
				new_s[2:4] = s[[3, 2]]
				new_s[np.arange(5, self.state_len, 2)] = 1-s[np.arange(5, self.state_len, 2)]
			self.full_loss_indices[i] = loss_state_to_index[tuple(new_s)]

