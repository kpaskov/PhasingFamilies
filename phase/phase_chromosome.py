import sys
from os import listdir
from itertools import product


from inheritance_states import AutosomalInheritanceStates
from input_output import WGSData, write_to_file, pull_families
from transition_matrices import AutosomalTransitionMatrix
from genotypes import Genotypes
from losses import LazyLoss
from viterbi import viterbi_forward_sweep_autosomes, viterbi_backward_sweep_autosomes
from mask import mask_states

# Run locally with python3 phase/phase_chromosome.py 22 3 data/160826.ped split_gen_miss phased

if __name__ == "__main__":

	# Read in command line arguments
	chrom = sys.argv[1]
	m = int(sys.argv[2])
	ped_file = sys.argv[3]
	data_dir = sys.argv[4]
	out_dir = sys.argv[5]
	batch_size = None if len(sys.argv) < 8 else int(sys.argv[6])
	batch_num = None if len(sys.argv) < 8 else int(sys.argv[7])
	batch_offset = None

	#shift_costs = [10]*4 + [500]*(2*(m-2))
	shift_costs = [10]*4 + [10*(m-2)]*(2*(m-2))


	# set up filenames
	sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
	coord_file = '%s/chr.%s.gen.coordinates.npy' % (data_dir,  chrom)
	gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s' % chrom) in f and 'gen.npz' in f])

	fam_output_file = '%s/chr.%s.familysize.%s.families.txt' % (out_dir, chrom, m)
	phase_output_file = '%s/chr.%s.familysize.%s.phased.txt' % (out_dir, chrom, m)

	if batch_size is not None:
		batch_offset = batch_size*batch_num
		fam_output_file = fam_output_file[:-4] + str(batch_num) + '.txt'
		phase_output_file = phase_output_file[:-4] + str(batch_num) + '.txt'

	# pull families of interest
	families_of_this_size = pull_families(sample_file, ped_file, m, batch_size, batch_offset)

	# create inheritance states
	inheritance_states = AutosomalInheritanceStates(m)
	
	# create transition matrix
	transition_matrix = AutosomalTransitionMatrix(inheritance_states, shift_costs)

	# create genotypes
	genotypes = Genotypes(m)

	# create loss function
	loss = LazyLoss(m, inheritance_states, genotypes)

	# get ready to pull processed WGS data 
	wgs_data = WGSData(data_dir, gen_files, coord_file, sample_file, chrom)

	with open(fam_output_file, 'w+') as famf, open(phase_output_file, 'w+') as statef:
		# write headers
		famf.write('family_id\tmother_id\tfather_id\t' + '\t'.join(['child%d_id' % i for i in range(1, m-1)]) + '\n')
		statef.write('\t'.join(['family_id', 'state_id', 'm1_state', 'm2_state', 'p1_state', 'p2_state',
			'\t'.join(['child%d_%s_state' % ((i+1), c) for i, c in product(range(m-2), ['m', 'p'])]),
			'start_pos', 'end_pos', 'start_family_index', 'end_family_index' 'pos_length', 'family_index_length']) + '\n')

		# phase each family
		for fkey, inds in families_of_this_size:
			print('family', fkey)

			# pull genotype data for this family
			family_genotypes, family_snp_positions, mult_factor = wgs_data.pull_data_for_individuals(inds)

			# forward sweep
			v_cost = viterbi_forward_sweep_autosomes(family_genotypes, family_snp_positions, mult_factor, inheritance_states, transition_matrix, loss)

			# backward sweep
			final_states = viterbi_backward_sweep_autosomes(v_cost, inheritance_states, transition_matrix)

			# mask messy areas
			final_states = mask_states(family_genotypes, mult_factor, final_states, inheritance_states, loss)

			# write to file
			write_to_file(famf, statef, fkey, inds, final_states, family_snp_positions)
