import numpy as np
from itertools import product
from collections import Counter

# genotype (pred, obs): cost
#g_cost = {
#	(-1, -1): 0,
#	(-1, 0): 1,
#	(-1, 1): 1,
#	(-1, 2): 1,
#	(0, -1): 0,
#	(0, 0): 0,
#	(0, 1): 1,
#	(0, 2): 2,
#	(1, -1): 0,
#	(1, 0): 1,
#	(1, 1): 0,
#	(1, 2): 1,
#	(2, -1): 0,
#	(2, 0): 2,
#	(2, 1): 1,
#	(2, 2): 0
#}

# -1 = ./.
# 0 = 0/0
# 1 = 0/1
# 2 = 1/1
# 3 = -/0 (hemizygous ref)
# 4 = -/1 (hemizygous alt)
# 5 = -/- (double deletion)
# 6 = 0/0/0
# 7 = 0/0/1
# 8 = 0/1/1
# 9 = 1/1/1

# This code takes advantage of the fact that our loss is symmetric in the sense that the state (1, 1, 1, 1, 0, 1, 0, 1)
# will have equivalent losses for all genotypes as (1, 1, 1, 1, 1, 0, 1, 0). Essentially, we can swap parental
# chromosomes without a problem.

# We arbitrarily fix positions 4 and 5 (the chromosomal inheritance of child 1) to be 0, 0 when calculating our
# perfect matches and then we cache a mapping between equivalent states with the __build_loss_equivalence__ function.
# This mapping allows us to produce losses for all possible states after only explicitly calculating the losses
# for some of them. This trick reduces both compute and memory by a factor of >4. Since this is our most
# memory intensive piece of code, this is very useful.

class LazyLoss:
	def __init__(self, inheritance_states, genotypes, params):

		# pull params
		pred_value_to_param = {0: '0/0', 1: '0/1', 2: '1/1', 3: '-/0', 4: '-/1', 5: '-/-'}#, 6: '0/0/0', 7: '0/0/1', 8: '0/1/1', 9: '1/1/1'}
		obs_value_to_param = {-1: './.', 0: '0/0', 1: '0/1', 2: '1/1'}
		hts_mult = params['x-times higher probability of error in hard-to-sequence region']

		preds = sorted(pred_value_to_param.keys())
		obss = sorted(obs_value_to_param.keys())

		# typical region costs
		self.g_cost = dict()
		for pred, obs in product(preds, obss):
			self.g_cost[(pred, obs)] = params['-log10(P[obs=%s|true_gen=%s])' % (obs_value_to_param[obs], pred_value_to_param[pred])]

		pred_to_correct_obs = dict()
		for pred in preds:
			pred_to_correct_obs[pred] = sorted(obss, key=lambda x: np.inf if self.g_cost[(pred, x)]==0 else self.g_cost[(pred, x)])[0]
		print('Prediction to best obs', pred_to_correct_obs)

		# hard-to-sequence region costs
		self.hts_g_cost = dict()
		for pred, obs in product(preds, obss):
			c = params['-log10(P[obs=%s|true_gen=%s])' % (obs_value_to_param[obs], pred_value_to_param[pred])]
				
			if pred_to_correct_obs[pred] != obs and c - np.log10(hts_mult) > 0.5:
				self.hts_g_cost[(pred, obs)] = c - np.log10(hts_mult)
			else:
				if pred_to_correct_obs[pred] != obs:
					print("Couldn't apply hard-to-sequence region factor to (%d->%d), would have created P(%d->%d)=%0.2f." % (pred, obs, pred, obs, pow(10, -c+np.log10(hts_mult))))
				self.hts_g_cost[(pred, obs)] = c

		for pred in preds:
			p = 1 - sum([pow(10, -self.hts_g_cost[(pred, obs)]) for obs in obss if pred_to_correct_obs[pred] != obs and self.hts_g_cost[(pred, obs)] != 0])
			self.hts_g_cost[(pred, pred_to_correct_obs[pred])] = -np.log10(p)
		#print('renormalized', [(pred, self.hts_g_cost[(pred, pred_to_correct_obs[pred])]) for pred in preds])

		for pred in preds:
			self.g_cost[(pred, -2)] = min(self.g_cost[(pred, -1)], self.g_cost[(pred, 0)])
			self.hts_g_cost[(pred, -2)] = min(self.hts_g_cost[(pred, -1)], self.hts_g_cost[(pred, 0)])

			self.hts_g_cost[(pred, -1)] = 0

		assert np.all(np.asarray(list(self.g_cost.values()))>=0)
		assert np.all(np.asarray(list(self.hts_g_cost.values()))>=0)

		print('\t' + ('\t\t'.join(map(str, obss + [-2]))))
		for pred in preds:
			print(str(pred) + '\t' + '\t'.join(['%0.2f-%0.2f' % (self.g_cost[(pred, obs)], self.hts_g_cost[pred, obs]) for obs in obss + [-2]]))

		self.m = inheritance_states.m
		self.q, self.state_len = genotypes.q, inheritance_states.state_len
		self.genotypes = genotypes

		self.__build_loss_equivalence__(inheritance_states)
		self.already_calculated = np.zeros((self.q,), dtype=bool)

		self.__build_perfect_matches__()
		
		# temp vector used in __call__
		self.is_hts = np.asarray([x[-1]==1 for x in self.loss_states], dtype=bool)
		print('hts loss states', np.sum(self.is_hts))
		
		self.s = np.zeros((len(self.perfect_matches)+1, 3), dtype=float)

		self.s[-1, :] = np.inf
		self.__precompute_loss_for_all_hom_ref__()

	def __call__(self, gen): 
		gen_index = self.genotypes.index(gen)
		if not self.already_calculated[gen_index]:
			for i, pm in enumerate(self.perfect_matches):
				self.s[i, 0] = sum([self.g_cost[(pred, obs)] for pred, obs in zip(pm, gen)])
				self.s[i, 1] = sum([self.hts_g_cost[(pred, obs)] for pred, obs in zip(pm, gen)])
				    
			values = np.min(self.s[self.perfect_match_indices[~self.is_hts, :], 0], axis=1)
			hts_values = np.min(self.s[self.perfect_match_indices[self.is_hts, :], 1], axis=1)
			
			self.losses[~self.is_hts, gen_index] = values
			self.losses[self.is_hts, gen_index] = hts_values

			self.already_calculated[gen_index] = True
		return self.losses[self.rep_state_indices, gen_index]

	def __precompute_loss_for_all_hom_ref__(self):
		self.__call__((-2,)*self.m)
		gen_index2 = self.genotypes.index((-2,)*self.m)
		for gen_index, gen in enumerate(self.genotypes):
			if len([x for x in gen if x>0]) == 0:
				self.losses[:, gen_index] = self.losses[:, gen_index2]
				self.already_calculated[gen_index] = True

	def __build_loss_equivalence__(self, inheritance_states):
		# states are equivalent if they have the same cost for every possible genotype
		# swapping parental chromosome copies produces equivalent states (ex m1->m2, m2->m1)

		# first, map every state to its representative equivalent state
		state_to_rep = dict()
		for s in inheritance_states:
			rep_s = None
			if s[4] == 0 and s[5] == 0:
				rep_s = s.tolist()
			elif s[4] == 1 and s[5] == 0:
				# flip maternal chromosome copies
				rep_s = s.tolist()
				rep_s[0], rep_s[1] = s[1], s[0]
				for i in np.arange(4, self.state_len-1, 2):
					rep_s[i] = 1-s[i]
			elif s[4] == 0 and s[5] == 1:
				# flip paternal chromosome copies
				rep_s = s.tolist()
				rep_s[2], rep_s[3] = s[3], s[2]
				for i in np.arange(5, self.state_len-1, 2):
					rep_s[i] = 1-s[i]
			elif s[4] == 1 and s[5] == 1:
				# flip maternal and paternal chromosome copies
				rep_s = s.tolist()
				rep_s[0], rep_s[1], rep_s[2], rep_s[3] = s[1], s[0], s[3], s[2]
				rep_s[:4] = [s[1], s[0], s[3], s[2]]
				for i in np.arange(4, self.state_len-1, 1):
					rep_s[i] = 1-s[i]
			else:
				print('Error, no equivalent loss')

			state_to_rep[tuple(s)] = tuple(rep_s)

		# build a list of loss_states and an index mapping
		self.loss_states = sorted(set(state_to_rep.values()))
		self.loss_state_to_index = dict([(x, i) for i, x in enumerate(self.loss_states)])
		self.losses = np.zeros((len(self.loss_states), self.q), dtype=float)
		self.rep_state_indices = [self.loss_state_to_index[state_to_rep[tuple(s)]] for s in inheritance_states]

		print('losses', self.losses.shape)


	def __build_perfect_matches__(self):
		# every state has <=16 perfect match family genotypes
		# futhermore, many of these perfect match family genotypes overlap
		# we generate a list of all perfect matches (self.perfect_matches)
		# and then we generate a len(loss_states) x 16 matrix of indices indicating the perfect matches for each loss_state

		# -1 = ./.
		# 0 = 0/0
		# 1 = 0/1
		# 2 = 1/1
		# 3 = -/0 (hemizygous ref)
		# 4 = -/1 (hemizygous alt)
		# 5 = -/- (double deletion)
		# 6 = 0/0/0
		# 7 = 0/0/1
		# 8 = 0/1/1
		# 9 = 1/1/1

		state_to_perfect_matches = dict()
		mat_pat_to_gen = {
			('-', '-'): 5, ('-', '0'): 3, ('-', '1'): 4, #('-', '00'): 0, ('-', '01'): 1, ('-', '11'): 2,
			('0', '-'): 3, ('0', '0'): 0, ('0', '1'): 1, #('0', '00'): 6, ('0', '01'): 7, ('0', '11'): 8,
			('1', '-'): 4, ('1', '0'): 1, ('1', '1'): 2, #('1', '00'): 7, ('1', '01'): 8, ('1', '11'): 9,
			#('00', '-'): 0, ('00', '0'): 6, ('00', '1'): 7,
			#('01', '-'): 1, ('01', '0'): 7, ('01', '1'): 8,
			#('11', '-'): 2, ('11', '0'): 8, ('11', '1'): 9
			('-', '2'): 1, ('2', '-'): 1,
			('0', '2'): 1, ('2', '0'): 1,
			('1', '2'): 1, ('2', '1'): 1,
			('2', '2'): 1

		}
		state_to_options = {0: ['-'], 1: ['0', '1', '2']}

		for s in self.loss_states:
			anc_pos = [state_to_options[x] for x in s[:4]]
			anc_variants = list(product(*anc_pos))

			state_to_perfect_matches[s] = []
			for av in anc_variants:
				pm = [mat_pat_to_gen[tuple(av[:2])], mat_pat_to_gen[tuple(av[2:4])]] # parents
				for i in range(self.m-2): # children
					mat, pat = s[(4+(2*i)):(6+(2*i))]
					pm.append(mat_pat_to_gen[(av[mat], av[2+pat])])
				state_to_perfect_matches[s].append(tuple(pm))

		self.perfect_matches = sorted(set(sum(state_to_perfect_matches.values(), [])))
		print('perfect matches', len(self.perfect_matches))

		perfect_match_to_index = dict([(x, i) for i, x in enumerate(self.perfect_matches)])
		self.perfect_match_indices = [[perfect_match_to_index[pm] for pm in state_to_perfect_matches[s]] for s in self.loss_states]

		# perfect_match_indices is ragged, square it off
		max_match = max([len(x) for x in self.perfect_match_indices])
		for pm in self.perfect_match_indices:
			while len(pm) < max_match:
				pm.append(len(perfect_match_to_index))
		self.perfect_match_indices = np.asarray(self.perfect_match_indices, dtype=int)

		print('perfect_match_indices', self.perfect_match_indices.shape)

	# def get_parental_variants(self, state, gen):
	# 	if (state, gen) in self.parental_variants:
	# 		# these are perfect matches - huge majority of cases we encounter
	# 		return self.parental_variants[(state, gen)], 0, tuple((0,)*self.m)
	# 	else:
	# 		# these are imperfect, we calculate on the fly
	# 		gen_index = self.genotypes.index(gen)
	# 		new_state_index = self.full_loss_indices[self.inheritance_states.index(state)]
	# 		s = self.loss_states[new_state_index]

	# 		# will we need to flip mom or dad?
	# 		flip_mom = (state[4] == 1)
	# 		flip_dad = (state[5] == 1)

	# 		min_value, min_indices, min_blame = None, [], []
	# 		for i, pm in enumerate(self.pm_gen[self.pm_gen_indices[new_state_index, :]]):
	# 			blame = [g_cost[(pred, obs)] for pred, obs in zip(pm, gen)]
	# 			v = sum(blame)
	# 			if min_value is None or v < min_value:
	# 				min_value = v
	# 				min_indices = [i]
	# 				min_blame = [blame]
	# 			elif v == min_value:
	# 				min_indices.append(i)
	# 				min_blame.append(blame)

	# 		# calculate parental variants
	# 		anc_pos = [[-1, -1] if s[i] == 0 else [0, 1] for i in range(4)]
	# 		all_options = np.array(list(product(*anc_pos)), dtype=np.int8)[min_indices, :]
	# 		pv = -np.ones((4,), dtype=np.int8)
	# 		pv[s[:4]==0] = -2
	# 		pv[np.all(all_options==0, axis=0)] = 0
	# 		pv[np.all(all_options==1, axis=0)] = 1

	# 		if flip_mom:
	# 			pv = pv[[1, 0, 2, 3]]
	# 		if flip_dad:
	# 			pv = pv[[0, 1, 3, 2]]

	# 		# distribute blame
	# 		if min_value == 0:
	# 			blame = (0.0,)*self.m
	# 		else:
	# 			blame = np.sum(np.asarray(min_blame), axis=0)
	# 			blame = min_value*blame/np.sum(blame) # normalize

	# 		return tuple(pv), int(min_value), tuple(blame)

