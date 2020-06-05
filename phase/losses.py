import numpy as np
from itertools import product
from collections import Counter

# -1 = ./.
# 0 = 0/0
# 1 = 0/1
# 2 = 1/1
# 3 = -/0 (hemizygous ref)
# 4 = -/1 (hemizygous alt)
# 5 = -/- (double deletion)


class LazyLoss:
	def __init__(self, states, genotypes, famkey, inds, params):

		# pull params
		pred_value_to_param = {0: '0/0', 1: '0/1', 2: '1/1', 3: '-/0', 4: '-/1', 5: '-/-'}
		obs_value_to_param = {-1: './.', 0: '0/0', 1: '0/1', 2: '1/1'}
		hts_mult = params['x-times higher probability of error in hard-to-sequence region']

		preds = sorted(pred_value_to_param.keys())
		obss = sorted(obs_value_to_param.keys())

		# typical region costs
		self.g_cost = dict()
		self.hts_g_cost = dict()

		emission_params = np.zeros((len(inds), len(preds)*len(obss)))
		for i, (pred, obs) in enumerate(product(preds, obss)):
			for j, ind in enumerate(inds):
				if (famkey + '.' + ind) in params:
					emission_params[j, i] = params[famkey + '.' + ind]['-log10(P[obs=%s|true_gen=%s])' % (obs_value_to_param[obs], pred_value_to_param[pred])]
				else:
					segments = famkey.split('.')
					if ('.'.join([segments[i] for i in [0, 3, 4]]) + '.' + ind) in params:
						emission_params[j, i] = params['.'.join([segments[i] for i in [0, 3, 4]]) + '.' + ind]['-log10(P[obs=%s|true_gen=%s])' % (obs_value_to_param[obs], pred_value_to_param[pred])]
					else:
						raise Exception('Cant find params for individual %s OR %s.' % (famkey + '.' + ind, '.'.join(famkey.split('.')[:3]) + '.' + ind))

		for k in range(emission_params.shape[1]):
			# if we can't estimate an error rate, use the mean value for everyone else
			emission_params[np.isnan(emission_params[:, k]), k] = np.nanmedian(emission_params[:, k])

			## if we can't estimate an error rate for anyone,
			# then an error will be raised


		for j, ind in enumerate(inds):
			# normal regions costs
			for i, (pred, obs) in enumerate(product(preds, obss)):
				self.g_cost[(j, pred, obs)] = emission_params[j, i]

			pred_to_correct_obs = dict()
			for pred in preds:
				pred_to_correct_obs[pred] = sorted(obss, key=lambda x: np.inf if self.g_cost[(j, pred, x)]==0 else self.g_cost[(j, pred, x)])[0]
			print('Prediction to best obs', pred_to_correct_obs)

			for pred, obs in pred_to_correct_obs.items():
				prob_of_error = sum([10.0**-self.g_cost[(j, pred, o)] for o in obss if o!=obs])
				self.g_cost[(j, pred, obs)] = -np.log10(1-prob_of_error)

			# hard-to-sequence region costs
			for pred, obs in product(preds, obss):
				if pred_to_correct_obs[pred] != obs:
					c = self.g_cost[(j, pred, obs)]
					if c > 2:
						self.hts_g_cost[(j, pred, obs)] = self.g_cost[(j, pred, obs)] - 1
					else:
						self.hts_g_cost[(j, pred, obs)] = self.g_cost[(j, pred, obs)]
			for pred, obs in pred_to_correct_obs.items():
				prob_of_error = sum([10.0**-self.hts_g_cost[(j, pred, o)] for o in obss if o!=obs])
				self.hts_g_cost[(j, pred, obs)] = -np.log10(1-prob_of_error)

			print(ind + '\t' + ('\t\t'.join(map(str, obss))))
			for pred in preds:
				print(str(pred) + '\t' + '\t'.join(['%0.4f-%0.4f' % (self.g_cost[(j, pred, obs)], self.hts_g_cost[(j, pred, obs)]) for obs in obss]))

		assert np.all(np.asarray(list(self.g_cost.values()))>=0)
		assert np.all(np.asarray(list(self.hts_g_cost.values()))>=0)

		self.q = genotypes.q
		self.genotypes = genotypes
		self.family_size = states.family_size

		self.losses = np.zeros((states.num_states, self.q), dtype=float)
		self.already_calculated = np.zeros((self.q,), dtype=bool)
		print('losses', self.losses.shape)

		self.__build_perfect_matches__(states)
		
		# temp vector used in __call__
		self.is_hts = np.asarray([x.is_hard_to_sequence() for x in states], dtype=bool)
		print('hts loss states', np.sum(self.is_hts))
		
		self.s = np.zeros((len(self.perfect_matches)+1, 2), dtype=float)

		self.__setup_ref_cost__()


	def __setup_ref_cost__(self):
		gen = (0,)*self.family_size
		gen_index = self.genotypes.index(gen)
		
		for i, pm in enumerate(self.perfect_matches):
			for gen in list(product(*[[0, -1]]*self.family_size)):
				self.s[i, 0] += 10**-sum([self.g_cost[(j, pred, obs)] for j, (pred, obs) in enumerate(zip(pm, gen))])
				self.s[i, 1] += 10**-sum([self.hts_g_cost[(j, pred, obs)] for j, (pred, obs) in enumerate(zip(pm, gen))])
		assert np.all(self.s<=1) and np.all(self.s>=0)
		self.s = -np.log10(self.s)
		
		values = np.min(self.s[self.perfect_match_indices, 0], axis=1)
		hts_values = np.min(self.s[self.perfect_match_indices[self.is_hts, :], 1], axis=1)
			
		self.losses[~self.is_hts, gen_index] += values[~self.is_hts]
		self.losses[self.is_hts, gen_index] += hts_values
		
		self.losses[:, gen_index] = 0
		self.already_calculated[gen_index] = True

	def __call__(self, gen): 
		gen_index = self.genotypes.index(gen)
		if not self.already_calculated[gen_index]:
			for i, pm in enumerate(self.perfect_matches):
				self.s[i, 0] = sum([self.g_cost[(j, pred, obs)] for j, (pred, obs) in enumerate(zip(pm, gen))])
				self.s[i, 1] = sum([self.hts_g_cost[(j, pred, obs)] for j, (pred, obs) in enumerate(zip(pm, gen))])
				    
			values = np.min(self.s[self.perfect_match_indices, 0], axis=1)
			hts_values = np.min(self.s[self.perfect_match_indices[self.is_hts, :], 1], axis=1)
			
			self.losses[~self.is_hts, gen_index] += values[~self.is_hts]
			self.losses[self.is_hts, gen_index] += hts_values

			self.already_calculated[gen_index] = True
		return self.losses[:, gen_index]


	def __build_perfect_matches__(self, states):
		# every state has <=16 perfect match family genotypes
		# futhermore, many of these perfect match family genotypes overlap
		# we generate a list of all perfect matches (self.perfect_matches)
		# and then we generate a len(states) x 16 matrix of indices indicating the perfect matches for each state

		# -1 = ./.
		# 0 = 0/0
		# 1 = 0/1
		# 2 = 1/1
		# 3 = -/0 (hemizygous ref)
		# 4 = -/1 (hemizygous alt)
		# 5 = -/- (double deletion)

		state_to_perfect_matches = dict()
		mat_pat_to_gen = {
			('-', '-'): 5, ('-', '0'): 3, ('-', '1'): 4,
			('0', '-'): 3, ('0', '0'): 0, ('0', '1'): 1,
			('1', '-'): 4, ('1', '0'): 1, ('1', '1'): 2,
			('-', '2'): 1, ('2', '-'): 1,
			('0', '2'): 1, ('2', '0'): 1,
			('1', '2'): 1, ('2', '1'): 1,
			('2', '2'): 1

		}
		state_to_options = {0: ['-'], 1: ['0', '1']}

		for s in states:
			anc_pos = [['-'] if s.has_deletion(i) else ['0', '1'] for i in range(4)]
			anc_variants = list(product(*anc_pos))

			state_to_perfect_matches[s] = []
			for av in anc_variants:
				#pm = [mat_pat_to_gen[tuple(av[:2])], mat_pat_to_gen[tuple(av[2:4])]] # parents
				pm = []
				for individual_index in range(self.family_size):
					mat, pat = s.get_phase(individual_index)
					pm.append(mat_pat_to_gen[(av[mat], av[pat])])
				state_to_perfect_matches[s].append(tuple(pm))

		self.perfect_matches = sorted(set(sum(state_to_perfect_matches.values(), [])))
		print('perfect matches', len(self.perfect_matches))

		perfect_match_to_index = dict([(x, i) for i, x in enumerate(self.perfect_matches)])
		self.perfect_match_indices = [[perfect_match_to_index[pm] for pm in state_to_perfect_matches[s]] for s in states]

		# perfect_match_indices is ragged, square it off
		max_match = max([len(x) for x in self.perfect_match_indices])
		for pm in self.perfect_match_indices:
			while len(pm) < max_match:
				pm.append(len(perfect_match_to_index))
		self.perfect_match_indices = np.asarray(self.perfect_match_indices, dtype=int)

		print('perfect_match_indices', self.perfect_match_indices.shape)


