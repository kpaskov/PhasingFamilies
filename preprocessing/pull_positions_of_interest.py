import numpy as np
from shutil import copyfile
from scipy import sparse
from os import listdir
import sys

data_dir = sys.argv[1] #'../split_gen_ihart'
pos_file = sys.argv[2] #'../data/23andme_positions.txt'
out_dir = sys.argv[3] #'../split_gen_ihart_23andme'

chroms = [str(x) for x in range(6, 7)]# + ['X', 'Y']

for chrom in chroms:
	print(chrom)
	# copy sample file to new directory
	copyfile('%s/chr.%s.gen.samples.txt' % (data_dir, chrom), '%s/chr.%s.gen.samples.txt' % (out_dir, chrom))

	# read in positions of interest
	pos_of_interest = list()
	with open(pos_file, 'r') as f:
	    next(f) # skip header
	    for line in f:
	        pieces = line.strip().split('\t')
	        if pieces[1] == chrom or (chrom == 'X' and pieces[1] == '23') or (chrom == 'Y' and pieces[1] == '24'):
	            pos_of_interest.append(int(pieces[2]))
	print('Positions of interest', len(set(pos_of_interest)))

	# copy positions
	coordinates = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir,  chrom))
	indices = np.isin(coordinates[:, 1], pos_of_interest) # must be in our list
	print('Overlapping positions', np.sum(indices))
	print('Missing positions', len(pos_of_interest)-np.sum(indices))

	np.save('%s/chr.%s.gen.coordinates.npy' % (out_dir, chrom), coordinates[indices, :])

	# pull genotype data from .npz
	gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])
	whole_chrom = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

	# pull out only positions of interest
	whole_chrom = whole_chrom[:, indices]
	print('Final data', whole_chrom.shape)
	sparse.save_npz('%s/chr.%s.gen' % (out_dir, chrom), whole_chrom)

	# pull pass info
	is_pass = np.load('%s/chr.%s.pass.npy' % (data_dir,  chrom))
	np.save('%s/chr.%s.pass.npy' % (out_dir, chrom), is_pass[indices])


