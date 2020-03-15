import sys
import json
import numpy as np

param_file = sys.argv[1]

genome_size = 3000000000

with open(param_file, 'r') as f:
	params = json.load(f)

# ------------------------------------ Estimate Rates of Other Events ------------------------------------

# estimate probability of recombination
mat_crossover = -np.log10(22.8)+np.log10(genome_size)
pat_crossover = -np.log10(1.7*22.8)+np.log10(genome_size)

num_deletions = 100
del_trans = -np.log10(2*num_deletions)+np.log10(genome_size)

num_hts = 1000
hts_trans = -np.log10(2*num_hts)+np.log10(genome_size)

params['-log10(P[inherited_deletion_entry_exit])'] = del_trans
params['-log10(P[denovo_deletion_entry_exit])'] = 9.15
params['-log10(P[maternal_crossover])'] = mat_crossover
params['-log10(P[paternal_crossover])'] = pat_crossover
params['-log10(P[hard_to_seq_region_entry_exit])'] = hts_trans
params['-log10(P[low_coverage_region_entry_exit])'] = hts_trans
params['x-times higher probability of error in hard-to-sequence region'] = 10

# for each individual, extrapolate deletion costs
obss = ['0/0', '0/1', '1/1', './.']
for k, v in params.items():
	if isinstance(v, dict):
		for obs in obss:
			params[k]['-log10(P[obs=%s|true_gen=-/0])' % obs] = params[k]['-log10(P[obs=%s|true_gen=0/0])' % obs]
			params[k]['-log10(P[obs=%s|true_gen=-/1])' % obs] = params[k]['-log10(P[obs=%s|true_gen=1/1])' % obs]
		
		params[k]['-log10(P[obs=0/0|true_gen=-/-])'] = params[k]['-log10(P[obs=0/1|true_gen=1/1])']
		params[k]['-log10(P[obs=0/1|true_gen=-/-])'] = (params[k]['-log10(P[obs=0/0|true_gen=1/1])'] + params[k]['-log10(P[obs=1/1|true_gen=0/0])'])/2
		params[k]['-log10(P[obs=1/1|true_gen=-/-])'] = params[k]['-log10(P[obs=0/1|true_gen=0/0])']
		params[k]['-log10(P[obs=./.|true_gen=-/-])'] = -np.log10(1 - sum([10.0**-params[k]['-log10(P[obs=%s|true_gen=-/-])' % obs]  for obs in ['0/0', '0/1', '1/1']]))

with open(param_file[:-5] + '_ext.json', 'w+') as f:
	json.dump(params, f, indent=4)
