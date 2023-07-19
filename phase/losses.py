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
	def __init__(self, states, family, params, num_loss_regions):

		obss = ['0/0', '0/1', '1/1', './.']
		preds = ['0/0', '0/1', '1/1', '-/0', '-/1']#, '-/-']#, '01/0', '01/1', '01/01']

		# loss region, individual, pred, obs
		self.emission_params = np.zeros((num_loss_regions, len(family), len(preds), len(obss)))
		self.no_data = np.zeros((len(family),), dtype=bool)
		for k in range(num_loss_regions):
			for pred, obs in product(preds, obss):
				pred_index, obs_index = preds.index(pred), obss.index(obs)
				for j, ind in enumerate(family.individuals):
					if family.id + '.' + ind in params:
						self.emission_params[k, j, pred_index, obs_index] = params[family.id + '.' + ind]['-log10(P[obs=%s|true_gen=%s,loss=%d])' % (obs, pred, k)]
					elif ind in params:
						self.emission_params[k, j, pred_index, obs_index] = params[ind]['-log10(P[obs=%s|true_gen=%s,loss=%d])' % (obs, pred, k)]
					else:
						# no sequence data for this individual, meaning all entries will be 0/0
						self.emission_params[k, j, pred_index, obs_index] = 0 if obs=='0/0' else np.inf
						self.no_data[j] = True
		
		print('no data', [family.individuals[i] for i in np.where(self.no_data)[0]])
		total_prob = np.sum(np.power(10, -self.emission_params[:, :, :, :]), axis=3)
		print(total_prob)


		for j, ind in enumerate(family.individuals):
			print(ind + '\n\t' + ('\t'.join(map(str, obss))))
			for pred_index, pred in enumerate(preds):
				print(str(pred) + '\t' + '\t'.join(['-'.join(['%0.4f' % self.emission_params[(k, j, pred_index, obs_index)] for k in range(num_loss_regions)]) for obs_index in range(len(obss))]))

		assert np.all(np.isnan(total_prob[:, ~self.no_data, :]) | np.isclose(total_prob[:, ~self.no_data, :], 1, atol=0.001))
		assert np.all(np.isnan(self.emission_params[:, ~self.no_data, :, :]) | (self.emission_params[:, ~self.no_data, :, :]>=0))

		self.num_loss_regions = num_loss_regions
		self.family_size = len(family)
		self.states = states

		self.max_pms = 2**(2*family.num_ancestors())
		ac_to_index = dict([(tuple(x), i) for i, x in enumerate(product(*[[0, 1, 2]]*(2*family.num_ancestors())))])
		self.num_acs = len(ac_to_index)

		self.perfect_matches = np.zeros((self.states.num_states, self.max_pms, len(family)), dtype=int)
		self.include_pm = np.inf*np.ones((self.states.num_states, self.max_pms), dtype=float)
		self.acs = np.zeros((self.states.num_states, self.num_acs), dtype=bool)
		for i, s in enumerate(states):
			pm, acs = self.states.get_perfect_matches(s)
			self.perfect_matches[i, :len(pm), :] = pm
			self.include_pm[i, :len(pm)] = 0
			self.acs[i, [ac_to_index[tuple(ac)] for ac in acs]]  = True
		self.loss_region = np.transpose(np.tile(np.array([x.loss_region() for x in states], dtype=int), (self.max_pms , 4, 1)), axes=[2, 0, 1])
		self.ind_indices = np.tile(np.arange(self.family_size, dtype=int), (self.states.num_states, self.max_pms , 1))


		print('perfect matches', self.perfect_matches.shape)
		print('loss region', self.loss_region.shape)
		print('ind', self.ind_indices.shape)


	def set_cache(self, family_genotypes):
		famgens, counts = np.unique(family_genotypes, axis=1, return_counts=True)
		indices = np.argsort(counts)

		cache_size = np.argmax(np.flip(np.cumsum(counts[indices]), axis=0) < np.arange(counts.shape[0]))
		cached_genotypes = famgens[:, indices[-cache_size:]].T

		self.gen_to_index = dict([(tuple(x), i) for i, x in enumerate(cached_genotypes)])
		self.already_calculated = np.zeros((len(self.gen_to_index),), dtype=bool)
		self.losses = np.zeros((self.states.num_states, len(self.gen_to_index)))

		print('cached losses', self.losses.shape, 'already_calculated', np.sum(self.already_calculated))
		print('losses', self.losses.shape, self.losses.nbytes/10**6, 'MB')


	def __call__(self, gen): 
		# -------------- any changes must also be made in ancestral_variant_probabilities() ----------------------
		gen_index = self.gen_to_index.get(tuple(gen), None)
		if gen_index is not None and self.already_calculated[gen_index]:
			return self.losses[:, gen_index]
		elif np.all(gen <= 0):	
			return np.zeros((self.states.num_states,))
		else:
			#loss = np.log10(np.sum(self.include_pm, axis=1))-np.log10(np.sum((10**-np.sum(self.emission_params[self.loss_region, self.ind_indices, self.perfect_matches, \
			#								np.tile(gen.astype(int), (self.states.num_states, self.max_pms, 1))], axis=2))*self.include_pm, axis=1))
			
			loss = np.nanmin(np.sum(self.emission_params[self.loss_region, self.ind_indices, self.perfect_matches, \
											np.tile(gen.astype(int), (self.states.num_states, self.max_pms, 1))], axis=2)+self.include_pm, axis=1)
			assert np.all(loss>0)

			if gen_index is not None:
				self.losses[:, gen_index] = loss
				self.already_calculated[gen_index] = True
			return loss

	def get_ancestral_variants(self, state_index, gen):
		avs = np.zeros((self.num_acs,))
		avs[self.acs[state_index, :]] = 10**-np.sum(self.emission_params[self.loss_region[state_index, :, :], self.ind_indices[state_index, :, :], self.perfect_matches[state_index, :, :], \
											np.tile(gen.astype(int), (self.max_pms, 1))], axis=1)[~np.isinf(self.include_pm[state_index, :])]
		return avs


