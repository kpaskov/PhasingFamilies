import pickle
import numpy as np
from scipy import sparse
from os import listdir
import sys

coord_file = sys.argv[1]
data_dir = sys.argv[2]
output_file = sys.argv[3]

# Read in variant coordinates
coordinates = np.asarray([tuple(map(int, x.decode('UTF-8').split('-'))) for x in pickle.load(open(coord_file, 'rb'))], dtype=int)

order_indices = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
reverse_order_indices = np.argsort(order_indices)

submatrices = []
for chrom in range(1, 23):
	print(chrom)
	# pull coordinates for chromosome
	chrom_coords = np.load('%s/chr.%d.gen.coordinates.npy' %(data_dir, chrom))

	# pull only coordinates belonging to snps
	snp_indices = np.where(chrom_coords[:, 2]==1)[0]
	chrom_coords = chrom_coords[snp_indices, :]

	to_find = coordinates[order_indices, :]
	to_find = to_find[to_find[:, 0]==chrom, :]

	# find indices of the variants we're looking for on this chromosome (indices are given in snp_index coordinates)
	pull_indices = np.searchsorted(chrom_coords[:, 1], to_find[:, 1])

	# load genotype data from .npz files
	offset = 0
	snp_offset = 0
	gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])
	print(gen_files)
	for gen_file in gen_files:
		part_chrom = sparse.load_npz('%s/%s' % (data_dir, gen_file))
		_, n = part_chrom.shape
		part_chrom = part_chrom[:, snp_indices[(snp_indices>=offset) & (snp_indices < (offset+n))]-offset]
		_, snp_n = part_chrom.shape
		
		submatrices.append(part_chrom[:, pull_indices[(pull_indices>=snp_offset) & (pull_indices<(snp_offset+snp_n))]-snp_offset])
		print(submatrices[-1].shape)
		offset += n
		snp_offset += snp_n

ordered_full_matrix = sparse.hstack(submatrices)
print(ordered_full_matrix.shape)

#random_full_coords = ordered_full_coords[reverse_order_indices, :]
random_full_matrix = ordered_full_matrix[:, reverse_order_indices]
#print(ordered_full_matrix.shape)
#print(np.hstack((coordinates, random_full_coords)))

sparse.save_npz(output_file, random_full_matrix)