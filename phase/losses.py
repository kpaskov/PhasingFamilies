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
		preds = ['0/0', '0/1', '1/1', '-/0', '-/1', '-/-']

		# loss region, individual, pred, obs
		self.emission_params = np.zeros((num_loss_regions, len(family), len(preds), len(obss)))
		no_data = set()
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
						no_data.add(ind)
				# if we can't estimate an error rate, use the median value for everyone else
				self.emission_params[k, np.isnan(self.emission_params[k, :, pred_index, obs_index]), pred_index, obs_index] = np.nanmedian(self.emission_params[k, :, pred_index, obs_index])
		print('no data', no_data)

		for j, ind in enumerate(family.individuals):
			print(ind + '\t' + ('\t\t'.join(map(str, obss))))
			for pred_index, pred in enumerate(preds):
				print(str(pred) + '\t' + '\t'.join(['-'.join(['%0.2f' % self.emission_params[(k, j, pred_index, obs_index)] for k in range(num_loss_regions)]) for obs_index in range(len(obss))]))

		assert np.all(self.emission_params>=0)

		self.num_loss_regions = num_loss_regions
		self.family_size = len(family)
		self.states = states
		self.alt_costs = [x if x <= -np.log10(0.5) else af_boundaries[i-1] for i, x in enumerate(af_boundaries)] + [af_boundaries[-1]]
		self.ref_costs = [-np.log10(1-(10**-alt_cost)) for alt_cost in self.alt_costs]
		#print('alt costs', self.alt_costs)
		#print('ref costs', self.ref_costs)

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

		self.__setup_ref_cost__()

		self.num_cached = 0
		self.num_uncached = 0

	def set_cache(self, family_genotypes):
		famgens, counts = np.unique(family_genotypes, axis=1, return_counts=True)
		indices = np.argsort(counts)

		cache_size = np.argmax(np.flip(np.cumsum(counts[indices])) < np.arange(counts.shape[0]))
		cached_genotypes = famgens[:, indices[-cache_size:]].T
		print(cached_genotypes.shape)
		print(cached_genotypes)

		old_indices = np.asarray([self.gen_to_index.get(tuple(x), -1) for x in cached_genotypes] + [-1])

		self.losses = self.losses[:, old_indices]
		self.already_calculated = self.already_calculated[old_indices]
		self.gen_to_index = dict([(tuple(x), i) for i, x in enumerate(cached_genotypes)])
		assert self.already_calculated[-1] == False
		assert len(self.gen_to_index) == self.losses.shape[1]-1
		print('cached losses', self.losses.shape, 'already_calculated', np.sum(self.already_calculated))

	def __setup_ref_cost__(self):
		gen = np.zeros((self.family_size,), dtype=int)
		gen_index = self.gen_to_index[(0,)*(self.family_size+1)]

		#for gen in list(product(*[[0, -1]]*self.family_size)):
		for i, pm in enumerate(self.perfect_matches):
			self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size), list(pm), gen], axis=1)
					    
		for k in range(self.num_loss_regions):
			self.losses[self.loss_region==k, gen_index] = -np.log10(np.sum(np.power(10, -(self.s[k, self.perfect_match_indices[self.loss_region==k, :]] + \
																		10*self.perfect_match_alt_alleles[self.loss_region==k, :])), axis=1))
		
		self.already_calculated[gen_index] = True

	def __call__(self, gen): 
		gen_index = self.gen_to_index.get(tuple(gen), None)

		#if (self.num_cached + self.num_uncached) % 1000 == 0:
		#	print(self.num_cached, self.num_uncached)

		if gen_index is not None and self.already_calculated[gen_index]:
			self.num_cached += 1
			return self.losses[:, gen_index]
		else:
			loss = np.zeros((self.states.num_states,), dtype=float)
			for i, pm in enumerate(self.perfect_matches):
				self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size), list(pm), list(gen[:-1])], axis=1)
					    
			for k in range(self.num_loss_regions):
				loss[self.loss_region==k] = -np.log10(np.sum(np.power(10, -(self.s[k, self.perfect_match_indices[self.loss_region==k, :]] + \
																								 self.ref_costs[gen[-1]]*self.perfect_match_ref_alleles[self.loss_region==k, :] + \
																								 self.alt_costs[gen[-1]]*self.perfect_match_alt_alleles[self.loss_region==k, :])), axis=1))
			if gen_index is not None:
				self.losses[:, gen_index] = loss
				self.already_calculated[gen_index] = True

			if gen_index is None:
				self.num_uncached += 1

			return loss


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

		state_to_perfect_matches = dict([(s, self.states.get_perfect_matches(s)) for s in states])
		self.perfect_matches = sorted(reduce((lambda x, y: x | y), [set(x[0]) for x in state_to_perfect_matches.values()]))
		perfect_match_to_index = dict([(x, i) for i, x in enumerate(self.perfect_matches)])
		print('perfect matches', len(self.perfect_matches))

		self.perfect_match_indices = []
		self.perfect_match_ref_alleles = []
		self.perfect_match_alt_alleles = []
		for s in states:
			pms, allele_combinations = state_to_perfect_matches[s]
			self.perfect_match_indices.append([perfect_match_to_index[pm] for pm in pms])
			self.perfect_match_ref_alleles.append([len([x for x in ac if x==0]) for ac in allele_combinations])
			self.perfect_match_alt_alleles.append([len([x for x in ac if x==1]) for ac in allele_combinations])

		self.perfect_match_indices = np.array(self.perfect_match_indices)
		self.perfect_match_ref_alleles = np.array(self.perfect_match_ref_alleles)
		self.perfect_match_alt_alleles = np.array(self.perfect_match_alt_alleles)

		print('perfect_match_indices', self.perfect_match_indices.shape)


