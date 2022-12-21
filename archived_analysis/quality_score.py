import sys
import json
from itertools import product, combinations
import traceback
from os import listdir
import numpy as np
from collections import defaultdict

from inheritance_states import InheritanceStates
from input_output import write_to_file, pull_families, pull_families_missing_parent, pull_gen_data_for_individuals
from transition_matrices import TransitionMatrix
from losses import LazyLoss
from viterbi import viterbi_forward_sweep, viterbi_backward_sweep, viterbi_forward_sweep_low_memory, viterbi_backward_sweep_low_memory

# python phase/phase_chromosome.py 22 data/v34.vcf.ped split_gen_ihart 37 phased_ihart parameter_estimation/params/ihart_multiloss_params.json --detect_deletions --family AU0197 2

import argparse

parser = argparse.ArgumentParser(description='Calculate quality scores for deletions.')
parser.add_argument('phase_dir', type=str, help='Json file of deletions.')
args = parser.parse_args()

with open('%s/info.json' % args.phase_dir, 'r') as f:
	data = json.load(f)

data_dir = data['data_dir']
with open('%s/info.json' % data_dir, 'r') as f:
	assembly = json.load(f)['assembly']

chrom = data['chrom']
if chrom is not None:
	chroms = [chrom]
else:
	chroms = [str(x) for x in range(1, 23)] + ['X']

param_file = data['param_file']
with open(param_file, 'r') as f:
	params = json.load(f)

num_loss_regions = data['num_loss_regions']
ped_file = data['ped_file']


# --------------- pull families of interest ---------------
families = pull_families(ped_file)

# make sure at least one individual has genetic data
sample_file = '%s/samples.json' % data_dir
with open(sample_file, 'r') as f:
	sample_ids = set(json.load(f))

samples_not_in_sample_ids = set()
samples_not_in_params = set()

for family in families:
	not_in_sample_ids = [x for x in family.individuals if x not in sample_ids]
	not_in_params = [x for x in family.individuals if x not in params and '%s.%s' % (family.id, x) not in params]
	
	family.prune(list(set(not_in_sample_ids+not_in_params)))
	samples_not_in_sample_ids.update(not_in_sample_ids)
	samples_not_in_params.update(not_in_params)

print('samples not in sample_ids %d' % len(samples_not_in_sample_ids))
print('samples not in params %d' % len(samples_not_in_params))

families = [x for x in families if x.num_descendents()>0]
print(len(families), 'have genomic data and parameters')
print('Families of interest', len(families))

# this script currently only works on simple, nuclear families
families = dict([(x.id, x) for x in families if x.num_ancestors()==2 and len(x.ordered_couples)==1])

print('families', len(families))

with open('%s/deletions.json' % args.phase_dir, 'r') as f:
	deletions = json.load(f)
print('deletions', len(deletions))

fam_to_deletions = defaultdict(list)
for d in deletions:
	fam_to_deletions[d['family']].append(d)

# score each deletion
for i, (fam, dels) in enumerate(sorted(fam_to_deletions.items())):
	print('fam %d / %d' % (i, len(fam_to_deletions)), len(dels))
	family = families[fam]
	m = len(family)

	# deletion loss
	del_inheritance_states = InheritanceStates(family, True, False, num_loss_regions)
	del_loss = LazyLoss(del_inheritance_states, family, params, num_loss_regions)
	del_transition_matrix = TransitionMatrix(del_inheritance_states, params)
	print('deletion loss created')

	# no deletion loss
	nodel_inheritance_states = InheritanceStates(family, False, False, num_loss_regions)
	nodel_loss = LazyLoss(nodel_inheritance_states, family, params, num_loss_regions)
	nodel_transition_matrix = TransitionMatrix(nodel_inheritance_states, params)
	print('no-deletion loss created')

	for deletion in dels:
		try:
			family_genotypes, family_snp_positions, mult_factor = pull_gen_data_for_individuals(data_dir, assembly, deletion['chrom'], family.individuals,
				start_pos=deletion['start_pos'], end_pos=deletion['end_pos'])
			del_loss.set_cache(family_genotypes)
			nodel_loss.set_cache(family_genotypes)

			# forward sweep
			v_cost_del = viterbi_forward_sweep(family_genotypes, family_snp_positions, mult_factor, del_inheritance_states, del_transition_matrix, del_loss, allow_del_start=True)
			v_cost_nodel = viterbi_forward_sweep(family_genotypes, family_snp_positions, mult_factor, nodel_inheritance_states, nodel_transition_matrix, nodel_loss, allow_del_start=True)

			# backward sweep
			final_states_del = viterbi_backward_sweep(v_cost_del, del_inheritance_states, del_transition_matrix, allow_del_end=True)
			final_states_nodel = viterbi_backward_sweep(v_cost_nodel, nodel_inheritance_states, nodel_transition_matrix, allow_del_end=True)

			if deletion['is_mat']:
				assert np.all(np.min(final_states_del[:2, :]==0, axis=0) < 1)
			else:
				assert np.all(np.min(final_states_del[2:4, :]==0, axis=0) < 1)

			assert not np.any(final_states_nodel[:4, :] < 1)

			deletion['quality_score'] = float(np.min(v_cost_nodel[:, -1]) - np.min(v_cost_del[:, -1]))
			print('quality_score', deletion['quality_score'])


		except Exception: 
 			traceback.print_exc()

	with open('%s/deletions.json' % args.phase_dir, 'w+') as f:
		json.dump(deletions, f, indent=4)



