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
	def __init__(self, states, family, params, num_loss_regions, af_boundaries):

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
		print(af_boundaries)
		self.alt_costs = np.array(af_boundaries[af_boundaries > -np.log10(0.5)].tolist() + [-np.log10(0.5)] + af_boundaries[af_boundaries < -np.log10(0.5)].tolist())
		self.ref_costs = np.array([-np.log10(1-(10**-alt_cost)) for alt_cost in self.alt_costs])
		#self.miss_costs = np.array([-np.log10(0.5) for _ in self.alt_costs])
		print('alt costs', self.alt_costs)
		print('ref costs', self.ref_costs)
		#print('miss costs', self.miss_costs)

		self.losses = np.zeros((states.num_states, 2))
		self.already_calculated = np.zeros((2,), dtype=bool)
		self.gen_to_index = {
			(0,)*(self.family_size+1): 0
		}

		self.__build_perfect_matches__(states)
		
		self.loss_region = np.asarray([x.loss_region() for x in states], dtype=int)
		
		# temp vector used in __call__
		self.s = np.zeros((num_loss_regions, len(self.perfect_matches)), dtype=float)
		self.s[:, -1] = np.inf


	def set_cache(self, family_genotypes):
		famgens, counts = np.unique(family_genotypes, axis=1, return_counts=True)
		indices = np.argsort(counts)

		cache_size = np.argmax(np.flip(np.cumsum(counts[indices]), axis=0) < np.arange(counts.shape[0]))
		cached_genotypes = famgens[:, indices[-cache_size:]].T

		old_indices = np.asarray([self.gen_to_index.get(tuple(x), -1) for x in cached_genotypes] + [-1])

		self.losses = self.losses[:, old_indices]
		self.already_calculated = self.already_calculated[old_indices]
		self.gen_to_index = dict([(tuple(x), i) for i, x in enumerate(cached_genotypes)])
		assert self.already_calculated[-1] == False
		assert len(self.gen_to_index) == self.losses.shape[1]-1
		print('cached losses', self.losses.shape, 'already_calculated', np.sum(self.already_calculated))


	def __call__(self, gen): 
		# -------------- any changes must also be made in ancestral_variant_probabilities() ----------------------
		if np.all(gen[:-1] <= 0) & ~np.all(gen[:-1] == 0):
			raise Exception('Genotypes without variants must be (0,)')

		gen_index = self.gen_to_index.get(tuple(gen), None)
		if gen_index is not None and self.already_calculated[gen_index]:
			return self.losses[:, gen_index]
		else:
			loss = np.zeros((self.states.num_states,), dtype=float)

			af_region_index = gen[-1]
			
			#rc, ac = -np.log10(0.5), -np.log10(0.5)
			rc, ac = self.ref_costs[af_region_index], self.alt_costs[af_region_index]
			#if len([x for x in gen if x==1 or x==2])>0 and len([x for x in gen if x==1 or x==0])>0:
			#	#rc, ac = -np.log10(0.5), -np.log10(0.5)
			#	rc, ac = self.ref_costs[af_region_index], self.alt_costs[af_region_index]
			#else:
			#	rc, ac = 0, 0

			if np.all(gen[:-1]==0):
				gen_options = list(product(*([[0] if x else [0, -1] for x in self.no_data])))
			else:
				gen_options = [gen[:-1]]

			for gen in gen_options:
				non_missing_indices = np.where([x!=-1 for x in gen])[0]
				for i, pm in enumerate(self.perfect_matches):
					self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size)[non_missing_indices], [pm[i] for i in non_missing_indices], [gen[i] for i in non_missing_indices]], axis=1)

				for k in range(self.num_loss_regions):
					loss[self.loss_region==k] += np.sum(np.power(10, -self.s[k, self.perfect_match_indices[self.loss_region==k, :]] - \
														(self.perfect_match_allele_counts[:, self.loss_region==k, :].T @ np.array([rc, ac, 0, rc+rc, -np.log10(2)+rc+ac, ac+ac])).T) * \
														self.not_filler[self.loss_region==k, :], axis=1)
			loss = -np.log10(np.clip(loss, 0, 1)) # handles numerical instability

			if gen_index is not None:
				self.losses[:, gen_index] = loss
				self.already_calculated[gen_index] = True
			
			return loss

	def ancestral_variant_probabilities(self, gen, state_index):
		# -------------- any changes must also be made in __call__() ----------------------
		if np.all(gen[:-1] <= 0) & ~np.all(gen[:-1] == 0):
			raise Exception('Genotypes without variants must be (0,)')

		gen_index = self.gen_to_index.get(tuple(gen), None)
		
		af_region_index = gen[-1]
		if np.all(gen[:-1]==0):
			gen_options = list(product(*([[0] if x else [0, -1] for x in self.no_data])))
		else:
			gen_options = [gen[:-1]]

		loss_region_index = self.states[state_index].loss_region()

		ancvar_probs = np.zeros((self.perfect_match_indices.shape[1],), dtype=float)
		for gen in gen_options:
			non_missing_indices = np.where([x!=-1 for x in gen])[0]
			for i, pm in enumerate(self.perfect_matches):
				self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size)[non_missing_indices], [pm[i] for i in non_missing_indices], [gen[i] for i in non_missing_indices]], axis=1)

			ancvar_probs += np.power(10, -self.s[k, self.perfect_match_indices[self.loss_region==k, :]] - \
														(self.perfect_match_allele_counts[:, self.loss_region==k, :].T @ np.array([rc, ac, -np.log10(1), rc+rc, -np.log10(2)+rc+ac, ac+ac])).T) * \
														self.not_filler[self.loss_region==k, :]

		return np.clip(ancvar_probs, 0, 1) # handles numerical instability



	def __build_perfect_matches__(self, states):
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
		self.perfect_match_allele_counts = np.zeros((6, states.num_states, max_perfect_matches), dtype=int)
		self.not_filler = np.zeros((states.num_states, max_perfect_matches), dtype=int)

		for i, s in enumerate(states):
			pms, allele_combinations = state_to_perfect_matches[s]
			self.perfect_match_indices[i, :len(pms)] = [perfect_match_to_index[pm] for pm in pms]
			for j in range(6):
				self.perfect_match_allele_counts[j, i, :len(pms)] = [len([x for x in ac if x==j]) for ac in allele_combinations]
			self.not_filler[i, :len(pms)] = 1

		#self.perfect_match_indices = np.array(self.perfect_match_indices)
		#self.perfect_match_ref_alleles = np.array(self.perfect_match_ref_alleles)
		#self.perfect_match_alt_alleles = np.array(self.perfect_match_alt_alleles)
		#self.perfect_match_missing_alleles = np.array(self.perfect_match_missing_alleles)

		print('perfect_match_indices', self.perfect_match_indices.shape)


