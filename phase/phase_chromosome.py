import sys
import json
from itertools import product, combinations
import traceback
import os
import numpy as np

from inheritance_states import InheritanceStates
from input_output import write_to_file, pull_families, pull_families_missing_parent, pull_gen_data_for_individuals
from transition_matrices import TransitionMatrix, TransitionMatrixX
from losses import LazyLoss
from viterbi import viterbi_forward_sweep, viterbi_backward_sweep, viterbi_forward_sweep_X, viterbi_backward_sweep_X

# python phase/phase_chromosome.py 22 data/v34.vcf.ped split_gen_ihart 37 phased_ihart parameter_estimation/params/ihart_multiloss_params.json --detect_deletions --family AU0197 2

import argparse

genome_size = 3000000000

parser = argparse.ArgumentParser(description='Phase chromosome.')
parser.add_argument('ped_file', type=str, help='Ped file of family structure.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data in .npy format.')
parser.add_argument('sequencing_error_rates', type=str, nargs='+', help='Sequencing error rates for model.')
parser.add_argument('--detect_inherited_deletions', action='store_true', default=False, help='Detect inherited deletions while phasing.')
parser.add_argument('--detect_upd', action='store_true', default=False, help='Detect uniparental disomy while phasing. We detect only heterodisomy because isodisomy is essentially indistinguishable from a denovo deletion.')
parser.add_argument('--chrom', type=str, default=None, help='Phase a single chrom.')
parser.add_argument('--family_size', type=int, default=None, help='Size of family to phase.')
parser.add_argument('--family', type=str, default=None, help='Phase only this family.')
parser.add_argument('--batch_size', type=int, default=None, help='Restrict number of families to batch_size.')
parser.add_argument('--batch_num', type=int, default=0, help='To be used along with batch_size to restrict number of families. Will use families[(batch_num*batch_size):((batch_num+1)*batch_size)]')
parser.add_argument('--no_overwrite', action='store_true', default=False, help='No overwriting files if they already exist.')
parser.add_argument('--retain_order', action='store_true', default=False, help='Default is to randomize order of offspring. If you want to retain order, set this flag.')
parser.add_argument('--missing_parent', action='store_true', default=False, help='Phase families that are missing a parent.')
parser.add_argument('--no_restrictions_X', action='store_true', default=False, help='Remove restriction on paternal recombination in the non-PAR X.')
parser.add_argument('--inherited_deletion_entry_exit_cost', type=float, default=-np.log10(2*100)+np.log10(genome_size), help='-log10(P[inherited_deletion_entry_exit])')
parser.add_argument('--maternal_crossover_cost', type=float, default=-np.log10(42)+np.log10(genome_size), help='-log10(P[maternal_crossover])')
parser.add_argument('--paternal_crossover_cost', type=float, default=-np.log10(28)+np.log10(genome_size), help='-log10(P[paternal_crossover])')
parser.add_argument('--loss_transition_cost', type=float, default=-np.log10(2*1000)+np.log10(genome_size), help='-log10(P[loss_transition])')
parser.add_argument('--phase_name', type=str, default=None, help='Name for this phase attempt.')


args = parser.parse_args()

if args.chrom is not None:
	chroms = [args.chrom]
else:
	chroms = [str(x) for x in range(1, 23)] + ['X']

if args.detect_inherited_deletions:
	print('Detecting inherited deletions while phasing ...')

if args.detect_upd:
	print('Detecting UPD while phasing...')

with open('%s/genotypes/info.json' % args.data_dir, 'r') as f:
	assembly = json.load(f)['assembly']

print('assembly', assembly)

# create output directory if it doesn't exist
if args.phase_name is None:
	out_dir = '%s/phase' % args.data_dir
else:
	out_dir = '%s/%s_phase' % (args.data_dir, args.phase_name)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open('%s/info.json' % out_dir, 'w+') as f:
	json.dump(vars(args), f)

# ---------------- set up sequencing error parameters ---------------------

params = dict()

all_params = []
for i, param_file in enumerate(args.sequencing_error_rates):
	with open(param_file, 'r') as f:
		all_params.append(json.load(f))
		params['loss_%d' % i] = param_file

# first pull individuals that exist in all params
individuals = set([k for k, v in all_params[0].items() if isinstance(v, dict)])
for p in all_params[1:]:
	individuals = individuals & set([k for k, v in p.items() if isinstance(v, dict)])

# for each individual, extrapolate deletion costs
gens = ['0/0', '0/1', '1/1']
obss = ['0/0', '0/1', '1/1', './.']

for ind in individuals:
	params[ind] = dict()
	for i, p in enumerate(all_params):
		for obs in obss:
			for gen in gens:
				params[ind]['-log10(P[obs=%s|true_gen=%s,loss=%d])' % (obs, gen, i)] = min(p[ind]['-log10(P[obs=%s|true_gen=%s])' % (obs, gen)], p[ind]['lower_bound[-log10(P[obs=%s|true_gen=%s])]' % (obs, gen)])
			params[ind]['-log10(P[obs=%s|true_gen=-/0,loss=%d])' % (obs, i)] = min(p[ind]['-log10(P[obs=%s|true_gen=0/0])' % obs], p[ind]['lower_bound[-log10(P[obs=%s|true_gen=0/0])]' % obs])
			params[ind]['-log10(P[obs=%s|true_gen=-/1,loss=%d])' % (obs, i)] = min(p[ind]['-log10(P[obs=%s|true_gen=1/1])' % obs], p[ind]['lower_bound[-log10(P[obs=%s|true_gen=1/1])]' % obs])
		
		params[ind]['-log10(P[obs=0/0|true_gen=-/-,loss=%d])' % i] = min(p[ind]['-log10(P[obs=0/1|true_gen=1/1])'], p[ind]['lower_bound[-log10(P[obs=0/1|true_gen=1/1])]'])
		params[ind]['-log10(P[obs=0/1|true_gen=-/-,loss=%d])' % i] = min((p[ind]['-log10(P[obs=0/0|true_gen=1/1])'] + p[ind]['-log10(P[obs=1/1|true_gen=0/0])'])/2, (p[ind]['lower_bound[-log10(P[obs=0/0|true_gen=1/1])]'] + p[ind]['lower_bound[-log10(P[obs=1/1|true_gen=0/0])]'])/2)
		params[ind]['-log10(P[obs=1/1|true_gen=-/-,loss=%d])' % i] = min(p[ind]['-log10(P[obs=0/1|true_gen=0/0])'], p[ind]['lower_bound[-log10(P[obs=0/1|true_gen=0/0])]'])
		params[ind]['-log10(P[obs=./.|true_gen=-/-,loss=%d])' % i] = -np.log10(1 - sum([10.0**-params[ind]['-log10(P[obs=%s|true_gen=-/-,loss=%d])' % (obs, i)]  for obs in ['0/0', '0/1', '1/1']]))


# --------------- pull families of interest ---------------
if args.missing_parent:
	families = pull_families_missing_parent(args.ped_file, args.data_dir, retain_order=args.retain_order)
else:
	families = pull_families(args.ped_file, retain_order=args.retain_order)

print(families[0], families[0].individuals)

# make sure at least one individual has genetic data
sample_file = '%s/genotypes/samples.json' % args.data_dir
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
		inheritance_states = InheritanceStates(family, args.detect_inherited_deletions, args.detect_inherited_deletions, args.detect_upd, len(args.sequencing_error_rates))
					
		# create transition matrix
		transition_matrix = TransitionMatrix(inheritance_states, 
			{
			'-log10(P[inherited_deletion_entry_exit])': args.inherited_deletion_entry_exit_cost,
			'-log10(P[maternal_crossover])': args.maternal_crossover_cost,
			'-log10(P[paternal_crossover])': args.paternal_crossover_cost,
			'-log10(P[loss_transition])': args.loss_transition_cost,
			})
		transitionsX = TransitionMatrixX(inheritance_states, 
			{
			'-log10(P[inherited_deletion_entry_exit])': args.inherited_deletion_entry_exit_cost,
			'-log10(P[maternal_crossover])': args.maternal_crossover_cost,
			'-log10(P[paternal_crossover])': args.paternal_crossover_cost,
			'-log10(P[loss_transition])': args.loss_transition_cost,
			})

		# create loss function for this family
		loss = LazyLoss(inheritance_states, family, params, len(args.sequencing_error_rates))

		# start by pulling header, or writing one if the file doesn't exist
		existing_phase_data = []
		needs_header = False
		try:
			with open('%s/inheritance_patterns/%s.phased.bed' % (out_dir, family), 'r') as f:
				next(f) # skip description (mom, dad, children)
				individuals = next(f).strip()[1:].split(',')
				family.set_individual_order(individuals)
				existing_phase_data = [x for x in f] 

		except (FileNotFoundError, StopIteration):
			header = '#mom, dad, children\n' + \
				'#' + (','.join(family.individuals)) + '\n' + \
				'#' + ('\t'.join(['chrom', 'chrom_start', 'chrom_end'] + \
				['m%d_del' % i for i in range(1, 2*len(family.mat_ancestors)+1)] + \
				['p%d_del' % i for i in range(1, 2*len(family.pat_ancestors)+1)] + \
				sum([['%s_mat' % x, '%s_pat' % x] for x in family.individuals], []) + \
				['loss_region'])) + '\n'
			needs_header = True

		# to avoid overwriting previous data, check which chromosomes have already been phased
		already_phased_chroms = set([line.split('\t', maxsplit=1)[0][3:] for line in existing_phase_data])
		print('already phased', sorted(already_phased_chroms))

		with open('%s/inheritance_patterns/%s.phased.bed' % (out_dir, family), 'a' if args.no_overwrite else 'w+') as statef:
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
				family_genotypes, family_snp_positions, mult_factor = pull_gen_data_for_individuals(args.data_dir, assembly, chrom, family.individuals)

				# update loss cache
				loss.set_cache(family_genotypes)

				if chrom == 'X' and (not args.no_restrictions_X):
					# forward sweep
					v_cost = viterbi_forward_sweep_X(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, transitionsX, loss, assembly)
					# backward sweep
					final_states, cost, ancestral_variants = viterbi_backward_sweep_X(v_cost, family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, transitionsX, loss, assembly)

				else:
					# forward sweep
					v_cost = viterbi_forward_sweep(family_genotypes, mult_factor, inheritance_states, transition_matrix, loss)
					# backward sweep
					final_states, cost, ancestral_variants = viterbi_backward_sweep(v_cost, family_genotypes, mult_factor, inheritance_states, transition_matrix, loss)


				#np.save('%s/%s.chr.%s.final_states' % (out_dir, family, chrom), final_states)
				#np.save('%s/%s.chr.%s.genomic_intervals' % (out_dir, family, chrom), family_snp_positions)
				#np.save('%s/%s.chr.%s.ancestral_variants' % (out_dir, family, chrom), ancestral_variants)
				#np.save('%s/%s.chr.%s.family_genotypes' % (out_dir, family, chrom), family_genotypes)
				

				# write to file
				write_to_file(statef, chrom, family, final_states, family_snp_positions, cost)

				statef.flush()
	except Exception: 
		traceback.print_exc()


	print('Done!')
	