This directory contains preliminary code for phasing nuclear families.

phase_chromosome.py <vcf_file> <ped_file> <out_directory> 
Phases all individuals in the vcf file (each family must have parents and at least 1 child). For now, families with more complex family structure (including grandparents or half siblings, etc) are ignored.
This script produces a series of .npz (compressed numpy arrays) files, one for each family.

The files are per family per chromosome and are in .npz file format (compressed numpy arrays).  Each .npz file can be loaded with np.load. Notation: m is the number of individuals in the family and n is the number of variants within that family. We only include variants for which the genotype of every family member is known and at least one family member is not homozygous reference. Family members are always ordered: mom, dad, child1, child2, etc.
	* X an mx4xn binary matrix which represents the phase and recombination points for each individual; Since this is a nuclear family, we have 4 ancestral chromosomes represented by the second dimension of X. Ancestral chromosomes are ordered maternal1, maternal2, paternal1, paternal2; so X[i, j, k] is 1 if individual i has ancestral chromosome j at position k and 0 if not. For example, the mother is at index i=0 so we have X[0, 0, :] = X[0, 1, :] = 1 and X[0, 2, :] = X[0, 3, :] = 0.
	* Y a 4xn binary matrix which indicates for each of the 4 ancestral chromosomes, whether the base is reference (0) or variant (1) at each location
	* data an mxn matrix of genotype data (0=hom ref, 1=het, 2 = hom alt)
	* row_indices a list of indices of all m individuals from the original vcf file
	* col_indices a list of indices of all variants from the original vcf file

aggregate_families.py <chromosome>
A script used to collect aggregate data across all families for a given chromosome.