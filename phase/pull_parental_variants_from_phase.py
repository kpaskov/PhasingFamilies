import sys
from os import listdir
import numpy as np
import scipy.sparse as sparse
import json

from inheritance_states import AutosomalInheritanceStates
from input_output import WGSData, pull_families_from_file, pull_phase, chrom_lengths, convert_to_csr
from genotypes import Genotypes
from losses import LazyLoss
from parental_variants import estimate_parental_variants

# Run locally with python3 phase/pull_parental_variants_from_phase.py 22 3 data/v34.vcf.ped split_gen_ihart phased_ihart parameter_estimation/ihart_params.json

if __name__ == "__main__":

	# Read in command line arguments
	chrom = sys.argv[1]
	m = int(sys.argv[2])
	ped_file = sys.argv[3]
	data_dir = sys.argv[4]
	phase_dir = sys.argv[5]
	param_file = sys.argv[6]

	with open(param_file, 'r') as f:
		params = json.load(f)

	print('Chromosome', chrom)
	chrom_length = chrom_lengths[chrom]

	# set up filenames
	sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
	coord_file = '%s/chr.%s.gen.coordinates.npy' % (data_dir,  chrom)
	gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s' % chrom) in f and 'gen.npz' in f])
	fam_file = '%s/chr.%s.familysize.%s.families.txt' % (phase_dir, chrom, m)
	phase_file = '%s/chr.%s.familysize.%s.phased.txt' % (phase_dir, chrom, m)

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



