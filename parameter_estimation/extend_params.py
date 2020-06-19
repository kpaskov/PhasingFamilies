import sys
import json
import numpy as np

param_files = sys.argv[1:-1]
out_file = sys.argv[-1]

genome_size = 3000000000

new_params = dict()

all_params = []
for i, param_file in enumerate(param_files):
	with open(param_file, 'r') as f:
		all_params.append(json.load(f))
		new_params['loss_%d' % i] = param_file

# ------------------------------------ Estimate Rates of Other Events ------------------------------------

# estimate probability of recombination
mat_crossover = -np.log10(22.8)+np.log10(genome_size)
pat_crossover = -np.log10(1.7*22.8)+np.log10(genome_size)

num_deletions = 100
del_trans = -np.log10(2*num_deletions)+np.log10(genome_size)

loss_trans = -np.log10(2*1000)+np.log10(genome_size)

new_params['-log10(P[inherited_deletion_entry_exit])'] = del_trans
new_params['-log10(P[denovo_deletion_entry_exit])'] = 9.15
new_params['-log10(P[maternal_crossover])'] = mat_crossover
new_params['-log10(P[paternal_crossover])'] = pat_crossover
new_params['-log10(P[loss_transition])'] = loss_trans

# first pull individuals that exist in all params
individuals = set([k for k, v in all_params[0].items() if isinstance(v, dict)])
for params in all_params[1:]:
	individuals = individuals & set([k for k, v in params.items() if isinstance(v, dict)])

# for each individual, extrapolate deletion costs
gens = ['0/0', '0/1', '1/1']
obss = ['0/0', '0/1', '1/1', './.']
num_overwritten = 0
for k in individuals:
	new_key = k
	if new_key in new_params:
		num_overwritten += 1
	new_params[new_key] = dict()
	for i, params in enumerate(all_params):
		for obs in obss:
			for gen in gens:
				new_params[new_key]['-log10(P[obs=%s|true_gen=%s,loss=%d])' % (obs, gen, i)] = params[k]['-log10(P[obs=%s|true_gen=%s])' % (obs, gen)]
			new_params[new_key]['-log10(P[obs=%s|true_gen=-/0,loss=%d])' % (obs, i)] = params[k]['-log10(P[obs=%s|true_gen=0/0])' % obs]
			new_params[new_key]['-log10(P[obs=%s|true_gen=-/1,loss=%d])' % (obs, i)] = params[k]['-log10(P[obs=%s|true_gen=1/1])' % obs]
		
		new_params[new_key]['-log10(P[obs=0/0|true_gen=-/-,loss=%d])' % i] = params[k]['-log10(P[obs=0/1|true_gen=1/1])']
		new_params[new_key]['-log10(P[obs=0/1|true_gen=-/-,loss=%d])' % i] = (params[k]['-log10(P[obs=0/0|true_gen=1/1])'] + params[k]['-log10(P[obs=1/1|true_gen=0/0])'])/2
		new_params[new_key]['-log10(P[obs=1/1|true_gen=-/-,loss=%d])' % i] = params[k]['-log10(P[obs=0/1|true_gen=0/0])']
		new_params[new_key]['-log10(P[obs=./.|true_gen=-/-,loss=%d])' % i] = -np.log10(1 - sum([10.0**-new_params[new_key]['-log10(P[obs=%s|true_gen=-/-,loss=%d])' % (obs, i)]  for obs in ['0/0', '0/1', '1/1']]))

with open(out_file, 'w+') as f:
	json.dump(new_params, f, indent=4)

print('Duplicates', num_overwritten)
