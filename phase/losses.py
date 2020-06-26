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
	def __init__(self, states, genotypes, family, params, num_loss_regions, af_boundaries):

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

		self.genotypes = genotypes
		self.num_loss_regions = num_loss_regions
		self.family_size = len(family)
		self.states = states
		self.af_boundaries = af_boundaries
		
		self.losses = np.zeros((states.num_states, len(genotypes)), dtype=float)
		self.already_calculated = np.zeros((len(genotypes),), dtype=bool)
		print('losses', self.losses.shape)

		self.__build_perfect_matches__(states)
		
		self.loss_region = np.asarray([x.loss_region() for x in states], dtype=int)
		
		# temp vector used in __call__
		self.s = np.zeros((num_loss_regions, len(self.perfect_matches)), dtype=float)
		self.s[:, -1] = np.inf

		self.__setup_ref_cost__()


	def __setup_ref_cost__(self):
		gen = (0,)*(self.family_size+1)
		gen_index = self.genotypes.index(gen)

		for gen in list(product(*[[0, -1]]*self.family_size)):
			alt_cost = 10 # just a big number (inf creates nans when multiplied by 0)
			ref_cost = 0
			l = np.zeros((self.losses.shape[0],))
			for i, pm in enumerate(self.perfect_matches):
				self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size), list(pm), list(gen)], axis=1)
					    
			for k in range(self.num_loss_regions):
				l[self.loss_region==k] = -np.log10(np.sum(np.power(10, -(self.s[k, self.perfect_match_indices[self.loss_region==k, :]] + \
																	    ref_cost*self.perfect_match_ref_alleles[self.loss_region==k, :] + \
																		alt_cost*self.perfect_match_alt_alleles[self.loss_region==k, :])), axis=1))
			if np.all(self.losses[:, gen_index]==0):
				self.losses[:, gen_index] = l
			else:
				self.losses[:, gen_index] = np.minimum(self.losses[:, gen_index], l)
		
		assert np.all(self.losses[:, gen_index] > 0)
		self.already_calculated[gen_index] = True

	def __call__(self, gen): 
		gen_index = self.genotypes.index(gen)
		alt_cost = self.af_boundaries[gen[-1]+1]
		ref_cost = -np.log10(1+np.log10(alt_cost))
		if not self.already_calculated[gen_index]:
			for i, pm in enumerate(self.perfect_matches):
				self.s[:, i] = np.sum(self.emission_params[:, np.arange(self.family_size), list(pm), list(gen[:-1])], axis=1)
				    
			for k in range(self.num_loss_regions):
				self.losses[self.loss_region==k, gen_index] = -np.log10(np.sum(np.power(10, -(self.s[k, self.perfect_match_indices[self.loss_region==k, :]] + \
																							 ref_cost*self.perfect_match_ref_alleles[self.loss_region==k, :] + \
																							 alt_cost*self.perfect_match_alt_alleles[self.loss_region==k, :])), axis=1))
			assert np.all(self.losses[:, gen_index] > 0)
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


