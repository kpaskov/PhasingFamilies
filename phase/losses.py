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
		preds = ['0/0', '0/1', '1/1', '-/0', '-/1', '-/-']#, '01/0', '01/1', '01/01']

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
				# if we can't estimate an error rate, use the median value from everyone else
				is_missing = np.isnan(self.emission_params[k, :, pred_index, obs_index])
				self.emission_params[k, is_missing & ~self.no_data, pred_index, obs_index] = np.nanmedian(self.emission_params[k, ~self.no_data, pred_index, obs_index])

			# for each individual and genotype, make sure emission probabilities sum to 1 (we may need to readjust if we filled in missing data)
			for j, pred_index in product(np.arange(len(family.individuals))[~self.no_data], range(len(preds))):
				home_obs = np.argmin(self.emission_params[k, j, pred_index, :])
				self.emission_params[k, j, pred_index, home_obs] = -np.log10(1-np.sum(np.power(10, -self.emission_params[k, j, pred_index, :]))+np.power(10, -self.emission_params[k, j, pred_index, home_obs]))
		print('no data', [family.individuals[i] for i in np.where(self.no_data)[0]])
		total_prob = np.sum(np.power(10, -self.emission_params), axis=3)
		assert np.all(np.isnan(total_prob[:, ~self.no_data, :]) | np.isclose(total_prob[:, ~self.no_data, :], 1))

		for j, ind in enumerate(family.individuals):
			print(ind + '\n\t' + ('\t'.join(map(str, obss))))
			for pred_index, pred in enumerate(preds):
				print(str(pred) + '\t' + '\t'.join(['-'.join(['%0.4f' % self.emission_params[(k, j, pred_index, obs_index)] for k in range(num_loss_regions)]) for obs_index in range(len(obss))]))

		assert np.all(np.isnan(self.emission_params[:, ~self.no_data, :, :]) | (self.emission_params[:, ~self.no_data, :, :]>=0))

		self.num_loss_regions = num_loss_regions
		self.family_size = len(family)
		self.states = states

		self.__build_perfect_matches__(states, family)
		
		self.loss_region = np.asarray([x.loss_region() for x in states], dtype=int)
		
		# temp vector used in __call__
		self.s = np.zeros((num_loss_regions, len(self.perfect_matches)), dtype=float)
		self.s[:, -1] = np.inf


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
		if np.all(gen <= 0) & ~np.all(gen == 0):
			raise Exception('Genotypes without variants must be (0,)')

		gen_index = self.gen_to_index.get(tuple(gen), None)
		if gen_index is not None and self.already_calculated[gen_index]:
			return self.losses[:, gen_index]
		elif np.any([x==-1 for x in gen]):
			# ignore positions where one or more family members has missing data
			loss = np.zeros((self.states.num_states,), dtype=float)
			if gen_index is not None:
				self.losses[:, gen_index] = loss
				self.already_calculated[gen_index] = True
			return loss
		else:
			loss = np.zeros((self.states.num_states,), dtype=float)


			if np.all(gen==0):
				gen_options = list(product(*([[0] if x else [0, -1] for x in self.no_data])))
			else:
				gen_options = [gen]

			for gen in gen_options:
				non_missing_indices = np.where([x!=-1 for x in gen])[0]
				for j, pm in enumerate(self.perfect_matches):
					self.s[:, j] = np.sum(self.emission_params[:, list(np.arange(self.family_size)[non_missing_indices]), [pm[i] for i in non_missing_indices], [gen[i] for i in non_missing_indices]], axis=1)

				for k in range(self.num_loss_regions):
					#loss[self.loss_region==k] += np.sum(np.power(10, -self.s[k, self.perfect_match_indices[self.loss_region==k, :]] - \
					#									(self.perfect_match_allele_counts[:, self.loss_region==k, :].T @ np.array([rc, ac, 0, rc+rc, -np.log10(2)+rc+ac, ac+ac])).T) * \
					#									self.not_filler[self.loss_region==k, :], axis=1)

					#print(np.max(np.power(10, -self.s[k, self.perfect_match_indices[self.loss_region==k, :]]) * self.not_filler[self.loss_region==k, :], axis=1))
					loss[self.loss_region==k] = np.maximum(loss[self.loss_region==k],
							np.max(np.power(10, -self.s[k, self.perfect_match_indices[self.loss_region==k, :]]) * self.not_filler[self.loss_region==k, :], axis=1))
			#loss = -np.log10(np.clip(loss, 0, 1)) # handles numerical instability
			loss = -np.log10(loss) # handles numerical instability
			#print(gen, loss)
			assert np.all(loss>=0)

			if gen_index is not None:
				self.losses[:, gen_index] = loss
				self.already_calculated[gen_index] = True
			return loss

	def get_ancestral_variants(self, state_index, gen):
		state = self.states[state_index]

		non_missing_indices = np.where([x!=-1 for x in gen])[0]

		anc_variants = np.zeros((self.num_acs,))
		anc_variants[self.ac_indices[state_index, self.not_filler[state_index, :]]] = [10**-np.sum(self.emission_params[state._loss_region, list(np.arange(self.family_size)[non_missing_indices]), [self.perfect_matches[pm_index][i] for i in non_missing_indices], [gen[i] for i in non_missing_indices]]) for pm_index in self.perfect_match_indices[state_index, self.not_filler[state_index, :]]]
		return anc_variants

	def __build_perfect_matches__(self, states, family):
		# every state has 2^(2*num_ancestors) perfect match family genotypes
		# futhermore, many of these perfect match family genotypes overlap
		# we generate a list of all perfect matches (self.perfect_matches)
		# and then we generate a len(states) x 2^(2*num_ancestors) matrix of indices indicating the perfect matches for each state

		# 0 = 0/0
		# 1 = 0/1
		# 2 = 1/1
		# 3 = -/0 (hemizygous ref)
		# 4 = -/1 (hemizygous alt)
		# 5 = -/- (double deletion)

		state_to_perfect_matches = dict([(s, self.states.get_perfect_matches(s)) for s in states])
		self.perfect_matches = sorted(reduce((lambda x, y: x | y), [set(x[0]) for x in state_to_perfect_matches.values()]))
		perfect_match_to_index = dict([(x, i) for i, x in enumerate(self.perfect_matches)])
		print('perfect matches', len(self.perfect_matches))

		max_perfect_matches = max([len(x[0]) for x in state_to_perfect_matches.values()])
		self.perfect_match_indices = np.zeros((states.num_states, max_perfect_matches), dtype=int)
		print('perfect_match_indices', self.perfect_match_indices.shape, self.perfect_match_indices.nbytes/10**6, 'MB')
		self.not_filler = np.zeros((states.num_states, max_perfect_matches), dtype=bool)
		print('not_filler', self.not_filler.shape, self.not_filler.nbytes/10**6, 'MB')

		ac_to_index = dict([(tuple(x), i) for i, x in enumerate(product(*[[0, 1, 2]]*(2*family.num_ancestors())))])
		self.ac_indices = np.zeros((states.num_states, max_perfect_matches), dtype=int)
		self.num_acs = len(ac_to_index)

		for i, s in enumerate(states):
			pms, allele_combinations = state_to_perfect_matches[s]
			self.perfect_match_indices[i, :len(pms)] = [perfect_match_to_index[pm] for pm in pms]
			self.ac_indices[i, :len(pms)] = [ac_to_index[tuple(x)] for x in allele_combinations]
			self.not_filler[i, :len(pms)] = True

