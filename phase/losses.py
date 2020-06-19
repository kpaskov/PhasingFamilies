import numpy as np
from itertools import product
from collections import Counter
from functools import reduce

# -1 = ./.
# 0 = 0/0
# 1 = 0/1
# 2 = 1/1
# 3 = -/0 (hemizygous ref)
# 4 = -/1 (hemizygous alt)
# 5 = -/- (double deletion)


class LazyLoss:
	def __init__(self, states, genotypes, family, params, num_loss_regions):

		obss = ['0/0', '0/1', '1/1', './.']
		preds = ['0/0', '0/1', '1/1', '-/0', '-/1', '-/-']

		# loss region, individual, pred, obs
		self.emission_params = np.zeros((num_loss_regions, len(family), len(preds), len(obss)))
		for k in range(num_loss_regions):
			for pred, obs in product(preds, obss):
				pred_index, obs_index = preds.index(pred), obss.index(obs)
				for j, ind in enumerate(family.individuals):
					if family.id + '.' + ind not in params:
						raise Exception('Cant find params for individual %s.' % ind)

					self.emission_params[k, j, pred_index, obs_index] = params[family.id + '.' + ind]['-log10(P[obs=%s|true_gen=%s,loss=%d])' % (obs, pred, k)]
			
				# if we can't estimate an error rate, use the median value for everyone else
				self.emission_params[k, np.isnan(self.emission_params[k, :, pred_index, obs_index]), pred_index, obs_index] = np.nanmedian(self.emission_params[k, :, pred_index, obs_index])

		for j, ind in enumerate(family.individuals):
			print(ind + '\t' + ('\t\t'.join(map(str, obss))))
			for pred_index, pred in enumerate(preds):
				print(str(pred) + '\t' + '\t'.join(['-'.join(['%0.2f' % self.emission_params[(k, j, pred_index, obs_index)] for k in range(num_loss_regions)]) for obs_index in range(len(obss))]))

		assert np.all(self.emission_params>=0)

		self.genotypes = genotypes
		self.num_loss_regions = num_loss_regions
		self.family_size = len(family)
		self.states = states
		
		self.losses = np.zeros((states.num_states, len(genotypes)), dtype=float)
		self.already_calculated = np.zeros((len(genotypes),), dtype=bool)
		print('losses', self.losses.shape)

		self.__build_perfect_matches__(states)
		
		self.loss_region = np.asarray([[x.loss_region()==k for x in states] for k in range(num_loss_regions)], dtype=bool)
		
		# temp vector used in __call__
		self.s = np.zeros((num_loss_regions, len(self.perfect_matches)+1), dtype=float)
		self.s[:, -1] = np.inf

		self.__setup_ref_cost__()


	def __setup_ref_cost__(self):
		gen = (0,)*self.family_size
		gen_index = self.genotypes.index(gen)
		
		for i, pm in enumerate(self.perfect_matches):
			self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size), list(pm), [0]*self.family_size], axis=1)
			for gen in list(product(*[[0, -1]]*self.family_size)):
				self.s[:, i] = np.minimum(self.s[:, i], np.sum(self.emission_params[:, np.arange(self.family_size), list(pm), list(gen)], axis=1))

		assert np.all(self.s[:, :-1]>0)

		for k in range(self.num_loss_regions):
			values = np.min(self.s[k, self.perfect_match_indices], axis=1)
			self.losses[self.loss_region[k, :], gen_index] += np.min(self.s[k, self.perfect_match_indices], axis=1)[self.loss_region[k, :]]
		self.already_calculated[gen_index] = True

	def __call__(self, gen): 
		gen_index = self.genotypes.index(gen)
		if not self.already_calculated[gen_index]:
			for i, pm in enumerate(self.perfect_matches):
				self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size), list(pm), list(gen)], axis=1)
				    
			for k in range(self.num_loss_regions):
				values = np.min(self.s[k, self.perfect_match_indices], axis=1)
				self.losses[self.loss_region[k, :], gen_index] += np.min(self.s[k, self.perfect_match_indices], axis=1)[self.loss_region[k, :]]

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

		state_to_perfect_matches = dict([(s, set(self.states.get_perfect_matches(s)[0])) for s in states])
		self.perfect_matches = sorted(reduce((lambda x, y: x | y), state_to_perfect_matches.values()))
		print('perfect matches', len(self.perfect_matches))

		perfect_match_to_index = dict([(x, i) for i, x in enumerate(self.perfect_matches)])
		self.perfect_match_indices = [[perfect_match_to_index[pm] for pm in sorted(state_to_perfect_matches[s])] for s in states]

		# perfect_match_indices is ragged, square it off
		max_match = max([len(x) for x in self.perfect_match_indices])
		for pm in self.perfect_match_indices:
			while len(pm) < max_match:
				pm.append(len(perfect_match_to_index))
		self.perfect_match_indices = np.asarray(self.perfect_match_indices, dtype=int)

		print('perfect_match_indices', self.perfect_match_indices.shape)


