import sys
import json
from itertools import product
import traceback
from os import listdir
import numpy as np

from inheritance_states import InheritanceStates
from input_output import write_to_file, pull_families, pull_gen_data_for_individuals
from transition_matrices import TransitionMatrix
from losses import LazyLoss
from viterbi import viterbi_forward_sweep, viterbi_backward_sweep

# python phase/phase_chromosome.py 22 data/v34.vcf.ped split_gen_ihart 37 phased_ihart parameter_estimation/params/ihart_multiloss_params.json --detect_deletions --family AU0197 2

import argparse

parser = argparse.ArgumentParser(description='Phase chromosome.')
parser.add_argument('ped_file', type=str, help='Ped file of family structure.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data in .npy format.')
parser.add_argument('assembly', type=str, help='Reference genome assembly for data.')
parser.add_argument('out_dir', type=str, help='Output directory.')
parser.add_argument('param_file', type=str, help='Parameters for model.')
parser.add_argument('num_loss_regions', type=int, help='Number of loss regions in model.')

parser.add_argument('--detect_deletions', action='store_true', default=False, help='Detect deletions while phasing.')
parser.add_argument('--family_size', type=int, default=None, help='Size of family to phase.')
parser.add_argument('--family', type=str, default=None, help='Phase only this family.')
parser.add_argument('--batch_size', type=int, default=None, help='Restrict number of families to batch_size.')
parser.add_argument('--batch_num', type=int, default=0, help='To be used along with batch_size to restrict number of families. Will use families[(batch_num*batch_size):((batch_num+1)*batch_size)]')
parser.add_argument('--no_overwrite', action='store_true', default=False, help='No overwriting files if they already exist.')
parser.add_argument('--detect_consanguinity', action='store_true', default=False, help='Detect consanguinity between parents. Can model basic consanguinity produced from a single shared ancestor. This option is only available for nuclear families.')

args = parser.parse_args()

chroms = [str(x) for x in range(1, 23)]

if args.detect_deletions:
	print('Detecting deletions while phasing ...')

if args.detect_consanguinity:
	print('Detecting parental consanguinity while phasing ...')

with open(args.param_file, 'r') as f:
	params = json.load(f)


# --------------- pull families of interest ---------------
families = pull_families(args.ped_file)

# make sure at least one individual has genetic data (chromosome 1 chosen arbitrarily)
sample_file = '%s/chr.%s.gen.samples.txt' % (args.data_dir, chroms[0])
with open(sample_file, 'r') as f:
	sample_ids = set([line.strip() for line in f])

for family in families:
	to_be_removed = [x for x in family.individuals if x not in sample_ids or (x not in params and '%s.%s' % (family.id, x) not in params)]
	family.prune(to_be_removed)

families = [x for x in families if x.num_descendents()>0]
print(len(families), 'have genomic data and parameters')

# set up af boundaries
num_unrelated_individuals = sum([f.num_ancestors() for f in families])
print('unrelated individuals in dataset', num_unrelated_individuals)

af_boundaries = np.arange(-np.log10(0.25), np.log10(2*num_unrelated_individuals/3), np.log10(2)).tolist() # use rule of 3; we can't differentiate allele frequencies any smaller than 3/(2*num unrelated individuals).
af_boundaries.extend([-np.log10(1-(10.0**-x)) for x in af_boundaries[1:]])
af_boundaries = np.array(sorted(af_boundaries, reverse=True))
print('af boundaries', af_boundaries)

# limit by size
if args.family_size is not None:
	families = [x for x in families if len(x) == args.family_size]

# limit to family
if args.family is not None:
	families = [x for x in families if x.id==args.family]

# no over-writing files
if args.no_overwrite:
	current_families = set([x[:-11] for x in listdir(args.out_dir) if x.endswith('.phased.txt')])
	families = [x for x in families if x.id not in current_families]

# limit to batch
if args.batch_size is not None:
	families = families[(args.batch_num*args.batch_size):((args.batch_num+1)*args.batch_size)]

# if we're modeling parental consanguinity, we can only work with nuclear families 
if args.detect_consanguinity:
	families = [x for x in families if x.num_ancestors()==2 and len(x.ordered_couples)==1]
	
	# to detect consanguinity, model parents as siblings - they have the freedom to inherit
	# completely different copies from "mat_shared_ancestor" and "pat_shared_ancestor"
	# or they can have consanguineous regions.
	for family in families:
		family.add_child(family.mat_ancestors[0], 'mat_shared_ancestor', 'pat_shared_ancestor')
		family.add_child(family.pat_ancestors[0], 'mat_shared_ancestor', 'pat_shared_ancestor')
print('Families of interest', len(families))

# phase each family
for family in families:
	try:
		print('family', family.id)

		# create inheritance states
		inheritance_states = InheritanceStates(family, args.detect_deletions, args.detect_deletions, args.num_loss_regions)
					
		# create transition matrix
		transition_matrix = TransitionMatrix(inheritance_states, params)

		# create loss function for this family
		loss = LazyLoss(inheritance_states, family, params, args.num_loss_regions, af_boundaries)
		#print('loss created')

		with open('%s/%s.phased.txt' % (args.out_dir, family), 'w+') as statef:
			# write header
			statef.write('\t'.join(['chrom'] + \
	                           ['m%d_del' % i for i in range(1, 2*len(family.mat_ancestors)+1)] + \
	                           ['p%d_del' % i for i in range(1, 2*len(family.pat_ancestors)+1)] + \
	                           sum([['%s_mat' % x, '%s_pat' % x] for x in family.individuals], []) + \
	                           ['loss_region', 'start_pos', 'end_pos']) + '\n')

			for chrom in chroms:
				print('chrom', chrom)

				# pull genotype data for this family
				family_genotypes, family_snp_positions, mult_factor = pull_gen_data_for_individuals(args.data_dir, af_boundaries, args.assembly, chrom, family.individuals)

				# update loss cache
				loss.set_cache(family_genotypes)

				# forward sweep
				v_cost = viterbi_forward_sweep(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, loss)

				# backward sweep
				final_states = viterbi_backward_sweep(v_cost, inheritance_states, transition_matrix)

				# write to file
				write_to_file(statef, chrom, family, final_states, family_snp_positions)

				statef.flush()
	except Exception: 
		traceback.print_exc()


	print('Done!')
	