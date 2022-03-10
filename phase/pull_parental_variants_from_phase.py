import sys
from os import listdir
import numpy as np
import scipy.sparse as sparse
import json

from inheritance_states import InheritanceStates
from input_output import write_to_file, pull_families, pull_families_missing_parent, pull_gen_data_for_individuals
from transition_matrices import TransitionMatrix
from losses import LazyLoss
from viterbi import viterbi_forward_sweep, viterbi_backward_sweep, viterbi_forward_sweep_low_memory, viterbi_backward_sweep_low_memory

import argparse

parser = argparse.ArgumentParser(description='Pull parental variants.')
parser.add_argument('phase_dir', type=str, help='Directory with phase data.')

args = parser.parse_args()

if __name__ == "__main__":

	with open('%s/info.json' % args.phase_dir, 'r') as f:
		info = json.load(f)

	with open(param_file, 'r') as f:
		params = json.load(f)

	with open('%s/info.json' % info['data_dir'], 'r') as f:
		assembly = json.load(f)['assembly']

	families = pull_families(args.ped_file, retain_order=info['retain_order'])

	# pull families
	families = pull_families_from_file(fam_file)

	# create inheritance states
	inheritance_states = AutosomalInheritanceStates(m)

	# create genotypes
	genotypes = Genotypes(m)

	# create loss function
	loss = LazyLoss(inheritance_states, genotypes, params)

	# get ready to pull processed WGS data 
	wgs_data = WGSData(data_dir, gen_files, coord_file, sample_file, ped_file, chrom)

	# pull variants for each family
	for fkey, inds in families:
		print('family', fkey)

		# pull genotype data for this family
		family_genotypes, family_snp_positions, mult_factor = wgs_data.pull_data_for_individuals(inds)

		# pull phase data for this family
		n = family_genotypes.shape[1]
		states = pull_phase(phase_file, fkey, m, n)

		# unmask regions of the parental chromosomes that haven't been inherited
		state_len = states.shape[0]
		maternal_indices = range(4, state_len, 2)
		paternal_indices = range(5, state_len, 2)

		m1_ninh = np.all(states[maternal_indices, :]!=0, axis=0)
		m2_ninh = np.all(states[maternal_indices, :]!=1, axis=0)
		p1_ninh = np.all(states[paternal_indices, :]!=0, axis=0)
		p2_ninh = np.all(states[paternal_indices, :]!=1, axis=0)

		states[0, m1_ninh] = 1
		states[1, m2_ninh] = 1
		states[2, p1_ninh] = 1
		states[3, p2_ninh] = 1

		# estimate variants
		parental_variants, cost, blame = estimate_parental_variants(loss, states, family_genotypes, chrom_length)

		# convert to parental_variants to column sparse matrices
		csr_parental_variants = convert_to_csr(parental_variants, family_snp_positions, mult_factor, chrom_length)
		csr_blame = convert_to_csr(blame, family_snp_positions, mult_factor, chrom_length)
		
		sparse.save_npz('%s/variants/chr.%s.%s.variants' % (phase_dir, chrom, fkey), csr_parental_variants)
		sparse.save_npz('%s/variants/chr.%s.%s.blame' % (phase_dir, chrom, fkey), csr_blame)



