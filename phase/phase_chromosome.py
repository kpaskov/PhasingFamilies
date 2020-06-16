import sys
import json
from os import listdir
from itertools import product


from inheritance_states import InheritanceStates
from input_output import WGSData, write_to_file, pull_families
from transition_matrices import TransitionMatrix
from genotypes import Genotypes
from losses import LazyLoss
from viterbi import viterbi_forward_sweep, viterbi_backward_sweep

# python phase/phase_chromosome.py 22 data/v34.vcf.ped split_gen_ihart 37 phased_ihart parameter_estimation/params/ihart_multiloss_params.json --detect_deletions --family AU0197 2

import argparse

parser = argparse.ArgumentParser(description='Phase chromosome.')
parser.add_argument('chrom', type=str, help='Chromosome.')
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
args = parser.parse_args()

if args.detect_deletions:
	print('Detecting deletions while phasing ...')

if args.chrom == '23':
	args.chrom = 'X'

with open(args.param_file, 'r') as f:
	params = json.load(f)

# --------------- set up filenames ---------------
sample_file = '%s/chr.%s.gen.samples.txt' % (args.data_dir, args.chrom)
coord_file = '%s/chr.%s.gen.coordinates.npy' % (args.data_dir,  args.chrom)
gen_files = sorted([f for f in listdir(args.data_dir) if ('chr.%s.' % args.chrom) in f and 'gen.npz' in f])

# --------------- pull families of interest ---------------
families = pull_families(sample_file, args.ped_file)

# limit by size
if args.family_size is not None:
	families = [x for x in families if len(x) == args.family_size]

# limit to family
if args.family is not None:
	families = [x for x in families if x.id==args.family]

# limit to batch
if args.batch_size is not None:
	families = families[(args.batch_num*args.batch_size):((args.batch_num+1)*args.batch_size)]

print('Families of interest', len(families))

# get ready to pull processed WGS data 
wgs_data = WGSData(args.data_dir, gen_files, coord_file, sample_file, args.ped_file, args.chrom, args.assembly)

# phase each family
for family in families:
	print('family', family.id)

	# create genotypes
	genotypes = Genotypes(len(family))

	# create inheritance states
	if args.chrom == 'X':
		inheritance_states = InheritanceStates(family, args.detect_deletions, True, args.num_loss_regions)
	else:
		inheritance_states = InheritanceStates(family, args.detect_deletions, args.detect_deletions, args.num_loss_regions)
				
	# create transition matrix
	transition_matrix = TransitionMatrix(inheritance_states, params)

	# pull genotype data for this family
	family_genotypes, family_snp_positions, mult_factor = wgs_data.pull_data_for_individuals(family.individuals)
	print('data pulled')

	# create loss function for this family
	loss = LazyLoss(inheritance_states, genotypes, family, params, args.num_loss_regions)
	print('loss created')

	# forward sweep
	v_cost = viterbi_forward_sweep(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, loss)
	print('forward sweep complete')

	# backward sweep
	final_states = viterbi_backward_sweep(v_cost, inheritance_states, transition_matrix)
	print('backward sweep complete')

	# write to file
	with open('%s/chr.%s.%s.phased.txt' % (args.out_dir, args.chrom, family), 'w+') as statef:
		write_to_file(statef, family, final_states, family_snp_positions)
	print('Done!')
	