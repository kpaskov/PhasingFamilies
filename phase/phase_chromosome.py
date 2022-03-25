import sys
import json
from itertools import product, combinations
import traceback
from os import listdir
import numpy as np

from inheritance_states import InheritanceStates
from input_output import write_to_file, pull_families, pull_families_missing_parent, pull_gen_data_for_individuals
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

parser.add_argument('--detect_inherited_deletions', action='store_true', default=False, help='Detect inherited deletions while phasing.')
parser.add_argument('--detect_upd', action='store_true', default=False, help='Detect uniparental disomy while phasing. We detect only heterodisomy because isodisomy is essentially indistinguishable from a denovo deletion.')
parser.add_argument('--chrom', type=str, default=None, help='Phase a single chrom.')
parser.add_argument('--family_size', type=int, default=None, help='Size of family to phase.')
parser.add_argument('--family', type=str, default=None, help='Phase only this family.')
parser.add_argument('--batch_size', type=int, default=None, help='Restrict number of families to batch_size.')
parser.add_argument('--batch_num', type=int, default=0, help='To be used along with batch_size to restrict number of families. Will use families[(batch_num*batch_size):((batch_num+1)*batch_size)]')
parser.add_argument('--no_overwrite', action='store_true', default=False, help='No overwriting files if they already exist.')
parser.add_argument('--retain_order', action='store_true', default=False, help='Default is to randomize order of offspring. If you want to retain order, set this flag.')
parser.add_argument('--low_memory', action='store_true', default=False, help='Reduce memory consumption, but no uncertainty regions.')
parser.add_argument('--missing_parent', action='store_true', default=False, help='Phase families that are missing a parent.')
parser.add_argument('--use_pass', action='store_true', default=False, help='If True, Use PASS filter to filter snps. If False, ignore PASS filter.')
parser.add_argument('--qs', action='store_true', default=False, help='Calculate quality score metrics.')

args = parser.parse_args()

if args.chrom is not None:
	chroms = [args.chrom]
else:
	chroms = [str(x) for x in range(1, 23)] + ['X']

if args.detect_inherited_deletions:
	print('Detecting inherited deletions while phasing ...')

if args.detect_upd:
	print('Detecting UPD while phasing...')

with open(args.param_file, 'r') as f:
	params = json.load(f)

with open('%s/info.json' % args.data_dir, 'r') as f:
	assembly = json.load(f)['assembly']

print('assembly', assembly)

with open('%s/info.json' % args.out_dir, 'w+') as f:
	json.dump(vars(args), f)


# --------------- pull families of interest ---------------
if args.missing_parent:
	families = pull_families_missing_parent(args.ped_file, args.data_dir, retain_order=args.retain_order)
else:
	families = pull_families(args.ped_file, retain_order=args.retain_order)

print(families[0], families[0].individuals)

# make sure at least one individual has genetic data
sample_file = '%s/samples.json' % args.data_dir
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

print('Families of interest, limited to batch', len(families))

# phase each family
for family in families:
	try:
		print('family', family.id)

		# create inheritance states
		inheritance_states = InheritanceStates(family, args.detect_inherited_deletions, args.detect_inherited_deletions, args.detect_upd, args.num_loss_regions)
					
		# create transition matrix
		transition_matrix = TransitionMatrix(inheritance_states, params)
		transition_matrixX = TransitionMatrixX(inheritance_states, params)

		# create loss function for this family
		loss = LazyLoss(inheritance_states, family, params, args.num_loss_regions)
		#print('loss created')

		if args.detect_inherited_deletions and args.qs:
			# same setup, but no mat deletions
			nomatdel_inheritance_states = InheritanceStates(family, False, True, args.num_loss_regions)
			nomatdel_transition_matrix = TransitionMatrix(nomatdel_inheritance_states, params)
			nomatdel_loss = LazyLoss(nomatdel_inheritance_states, family, params, args.num_loss_regions)

			# same setup, but no pat deletions
			nopatdel_inheritance_states = InheritanceStates(family, True, False, args.num_loss_regions)
			nopatdel_transition_matrix = TransitionMatrix(nopatdel_inheritance_states, params)
			nopatdel_loss = LazyLoss(nopatdel_inheritance_states, family, params, args.num_loss_regions)

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
				family_genotypes, family_snp_positions, mult_factor = pull_gen_data_for_individuals(args.data_dir, assembly, chrom, family.individuals, use_pass=args.use_pass)

				# update loss cache
				loss.set_cache(family_genotypes)

				if chrom == 'X':
					# forward sweep
					v_cost = viterbi_forward_sweep_X(family_genotypes, family_snp_positions, mult_factor, inheritance_states, inheritance_statesX, transition_matrix, loss, assembly)
					# backward sweep
					final_states, cost, ancestral_variants = viterbi_backward_sweep_X(v_cost, family_genotypes, family_snp_positions, mult_factor, inheritance_states, inheritance_statesX, transition_matrix, loss, assembly)

				else:
					# forward sweep
					v_cost = viterbi_forward_sweep(family_genotypes, mult_factor, inheritance_states, transition_matrix, loss)
					# backward sweep
					final_states, cost, ancestral_variants = viterbi_backward_sweep(v_cost, family_genotypes, mult_factor, inheritance_states, transition_matrix, loss)


				np.save('%s/%s.chr.%s.final_states' % (args.out_dir, family, chrom), final_states)
				np.save('%s/%s.chr.%s.genomic_intervals' % (args.out_dir, family, chrom), family_snp_positions)
				np.save('%s/%s.chr.%s.ancestral_variants' % (args.out_dir, family, chrom), ancestral_variants)
				np.save('%s/%s.chr.%s.family_genotypes' % (args.out_dir, family, chrom), family_genotypes)
				

				# write to file
				write_to_file(statef, chrom, family, final_states, family_snp_positions, cost)

				statef.flush()
	except Exception: 
		traceback.print_exc()


	print('Done!')
	