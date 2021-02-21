import sys
import json
from itertools import product, combinations
import traceback
from os import listdir
import numpy as np

from inheritance_states import InheritanceStates
from input_output import write_to_file, pull_families, pull_gen_data_for_individuals
from transition_matrices import TransitionMatrix
from losses import LazyLoss
from viterbi import viterbi_forward_sweep, viterbi_backward_sweep, viterbi_forward_sweep_low_memory, viterbi_backward_sweep_low_memory

# python phase/phase_chromosome.py 22 data/v34.vcf.ped split_gen_ihart 37 phased_ihart parameter_estimation/params/ihart_multiloss_params.json --detect_deletions --family AU0197 2

import argparse

parser = argparse.ArgumentParser(description='Phase chromosome.')
parser.add_argument('ped_file', type=str, help='Ped file of family structure.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data in .npy format.')
parser.add_argument('out_dir', type=str, help='Output directory.')
parser.add_argument('param_file', type=str, help='Parameters for model.')
parser.add_argument('num_loss_regions', type=int, help='Number of loss regions in model.')

parser.add_argument('--detect_deletions', action='store_true', default=False, help='Detect deletions while phasing.')
parser.add_argument('--detect_duplications', action='store_true', default=False, help='Detect duplications while phasing.')
parser.add_argument('--chrom', type=str, default=None, help='Phase a single chrom.')
parser.add_argument('--family_size', type=int, default=None, help='Size of family to phase.')
parser.add_argument('--family', type=str, default=None, help='Phase only this family.')
parser.add_argument('--batch_size', type=int, default=None, help='Restrict number of families to batch_size.')
parser.add_argument('--batch_num', type=int, default=0, help='To be used along with batch_size to restrict number of families. Will use families[(batch_num*batch_size):((batch_num+1)*batch_size)]')
parser.add_argument('--no_overwrite', action='store_true', default=False, help='No overwriting files if they already exist.')
parser.add_argument('--detect_consanguinity', action='store_true', default=False, help='Detect consanguinity between parents. Can model basic consanguinity produced from a single shared ancestor. This option is only available for nuclear families.')
parser.add_argument('--max_af_cost', type=float, default=np.log10(71702*2/3), help='Maximum allele frequency cost to consider. Should be set to something like np.log10(2*n/3) where n is the number of individuals used when estimating allele frequencies in data_dir/chr.*.gen.af.npy.')
parser.add_argument('--retain_order', action='store_true', default=False, help='Default is to randomize order of offspring. If you want to retain order, set this flag.')
parser.add_argument('--low_memory', action='store_true', default=False, help='Reduce memory consumption, but no uncertainty regions.')

args = parser.parse_args()

if args.chrom is not None:
	chroms = [args.chrom]
else:
	chroms = [str(x) for x in range(1, 23)] + ['X']

if args.detect_deletions:
	print('Detecting deletions while phasing ...')

if args.detect_consanguinity:
	print('Detecting parental consanguinity while phasing ...')

with open(args.param_file, 'r') as f:
	params = json.load(f)

with open('%s/info.json' % args.data_dir, 'r') as f:
	assembly = json.load(f)['assembly']

with open('%s/info.json' % args.out_dir, 'w+') as f:
	json.dump({
		'ped_file': args.ped_file,
		'data_dir': args.data_dir,
		'param_file': args.param_file,
		'num_loss_regions': args.num_loss_regions,
		'detect_deletions': args.detect_deletions,
		'chrom': args.chrom,
		'detect_consanguinity': args.detect_consanguinity,
		'max_af_cost': args.max_af_cost
		}, f)


# --------------- pull families of interest ---------------
families = pull_families(args.ped_file, retain_order=args.retain_order)

# make sure at least one individual has genetic data
sample_file = '%s/samples.json' % args.data_dir
with open(sample_file, 'r') as f:
	sample_ids = set(json.load(f))

for family in families:
	to_be_removed = [x for x in family.individuals if x not in sample_ids or (x not in params and '%s.%s' % (family.id, x) not in params)]
	family.prune(to_be_removed)

families = [x for x in families if x.num_descendents()>0]
print(len(families), 'have genomic data and parameters')

af_boundaries = np.arange(-np.log10(0.25), args.max_af_cost, np.log10(2)).tolist()
af_boundaries.extend([-np.log10(1-(10.0**-x)) for x in af_boundaries[1:]])
af_boundaries = np.array(sorted(af_boundaries, reverse=True))
print('af boundaries', af_boundaries)

# limit by size
if args.family_size is not None:
	families = [x for x in families if len(x) == args.family_size]

# limit to family
if args.family is not None:
	families = [x for x in families if x.id==args.family]

print('Families of interest', len(families))

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
print('Families of interest, limited to batch', len(families))

# phase each family
for family in families:
	try:
		print('family', family.id)

		# create inheritance states
		inheritance_states = InheritanceStates(family, args.detect_deletions, args.detect_deletions, 
												args.detect_duplications, args.detect_duplications, args.num_loss_regions)
					
		# create transition matrix
		transition_matrix = TransitionMatrix(inheritance_states, params)

		# create loss function for this family
		loss = LazyLoss(inheritance_states, family, params, args.num_loss_regions, af_boundaries)
		#print('loss created')

		# start by pulling header, or writing one if the file doesn't exist
		existing_phase_data = []
		needs_header = False
		try:
			with open('%s/%s.phased.txt' % (args.out_dir, family), 'r') as f:
				header = next(f)
				header_pieces = header.strip().split('\t')[1:-2]
				individuals = [x[:-4] for x in header_pieces if x.endswith('_mat')]
				family.set_individual_order(individuals)
				existing_phase_data = [x for x in f] 

		except (FileNotFoundError, StopIteration):
			header = '\t'.join(['chrom'] + \
				['m%d_del' % i for i in range(1, 2*len(family.mat_ancestors)+1)] + \
				['p%d_del' % i for i in range(1, 2*len(family.pat_ancestors)+1)] + \
				sum([['%s_mat' % x, '%s_pat' % x] for x in family.individuals], []) + \
				['loss_region', 'start_pos', 'end_pos']) + '\n'
			needs_header = True

		# to avoid overwriting previous data, check which chromosomes have already been phased
		already_phased_chroms = set([line.split('\t', maxsplit=1)[0][3:] for line in existing_phase_data])
		print('already phased', sorted(already_phased_chroms))

		with open('%s/%s.phased.txt' % (args.out_dir, family), 'a' if args.no_overwrite else 'w+') as statef:
			if args.no_overwrite:
				print('no overwrite')
				if needs_header:
					statef.write(header)
			else:
				print('overwriting', list(set(chroms) & already_phased_chroms))
				statef.write(header)
				for line in existing_phase_data:
					if line.split('\t', maxsplit=1)[0][3:] not in chroms:
						statef.write(line)
				already_phased_chroms = already_phased_chroms - set(chroms)

			for chrom in [x for x in chroms if x not in already_phased_chroms]:
				print('chrom', chrom)

				# pull genotype data for this family
				family_genotypes, family_snp_positions, mult_factor = pull_gen_data_for_individuals(args.data_dir, af_boundaries, assembly, chrom, family.individuals)

				# update loss cache
				loss.set_cache(family_genotypes)

				if args.low_memory:
					# forward sweep
					v_path, v_cost = viterbi_forward_sweep_low_memory(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, loss)

					# backward sweep
					final_states = viterbi_backward_sweep_low_memory(v_path, v_cost, inheritance_states, transition_matrix)

				else:
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
	