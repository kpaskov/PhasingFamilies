import numpy as np
from collections import namedtuple, defaultdict, Counter
from itertools import product, combinations
import json
import argparse
import input_output
import os
import traceback
from numpyencoder import NumpyEncoder

parser = argparse.ArgumentParser(description='Pull UPD from phasing output.')
parser.add_argument('dataset_name', type=str, help='Name of dataset.')

chroms_of_interest = [str(x) for x in range(1, 23)] #+ ['X']
args = parser.parse_args()


#pulls phase data from a file
def pull_phase(filename):
	with open(filename, 'r') as f:
		header = next(f).strip().split('\t')[1:-2] # skip header

		if len([x for x in header if x.endswith('_del')]) != 4:
			raise Exception('This is a complex family.')

		individuals = [x[:-4] for x in header if x.endswith('_mat')]

		if len(individuals) == 3:
			raise Exception('This is a trio.')

		states = []
		chrs = []
		starts = []
		ends = []
		for line in f:
			pieces = line.strip().split('\t')
			chrs.append(pieces[0][3:])
			states.append(list(map(int, pieces[1:-2])))
			starts.append(int(pieces[-2]))
			ends.append(int(pieces[-1]))

		mat_indices = [i for i, x in enumerate(header) if x.endswith('_mat')]
		pat_indices = [i for i, x in enumerate(header) if x.endswith('_pat')]

		states = np.array(states).T
		starts = np.array(starts)
		ends = np.array(ends)

		# if this is a hard to sequences region, we don't know the exact location of crossovers
		states[:, states[-1, :]==1] = -1


	return states, chrs, starts, ends, individuals, mat_indices, pat_indices
		

# extracts deletions from within upd interval
def extract_deletions(states, starts, ends, is_mat):
	#print(states)
	if is_mat:
		del_indices = [0, 1]
	else:
		del_indices = [2, 3]

	return np.any(states[del_indices, :-1]==0)


# extracts all UPD intervals from phase
UPD = namedtuple('UPD', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'family_size', 'mom', 'dad'])
def pull_upd(family_id, states, chroms, starts, ends, individuals,  indices, is_mat):

    def make_upd(chrom, start_index, end_index, individual):
    	return UPD(family_id, chrom, int(starts[start_index]), int(starts[end_index]),
                                    individual, not is_mat, is_mat, 
                                    len(individuals), individuals[0], individuals[1])

    upds = []
    for chrom in chroms_of_interest:
        # block out 
        is_chrom = np.array([x==chrom for x in chroms])
        for individual, index in zip(individuals[2:], indices[2:]):
            change_indices = np.where(is_chrom[:-1] & is_chrom[1:] & (states[index, :-1] != states[index, 1:]))[0]+1

            if change_indices.shape[0]>1:
                for current_index, next_index in zip(change_indices[:-1], change_indices[1:]):
                    assert states[index, current_index-1] != states[index, current_index]
                    assert states[index, current_index] != states[index, next_index]

                    if is_mat and (states[index, current_index] == 2 or states[index, current_index]==3):
                        upds.append(make_upd(chrom, current_index, next_index, individual))
                    elif (not is_mat) and (states[index, current_index] == 0 or states[index, current_index]==1):
                        upds.append(make_upd(chrom, current_index, next_index, individual))

    return upds

with open('%s/sibpairs.json' % args.dataset_name, 'r') as f:
	sibpairs = json.load(f)

all_upds = []
for sibpair in sibpairs:
	family_key = sibpair['family']
	print(family_key)
	states, chroms, starts, ends, individuals, mat_indices, pat_indices = pull_phase('%s/%s.phased.txt' % (sibpair['phase_dir'], family_key))
	
	missing_chroms = set(chroms_of_interest) - set(np.unique(chroms))
	if len(missing_chroms)>0:
		raise Exception('missing %s' % str(missing_chroms))

	mult = ends-starts
	num_children = len(individuals)-2

	# start by pulling all upds
	mat_upds = pull_upd(family_key, states, chroms, starts, ends, individuals, mat_indices, True)
	pat_upds = pull_upd(family_key, states, chroms, starts, ends, individuals, pat_indices, False)
	upds = mat_upds + pat_upds

	all_upds.extend(upds)

with open('%s/upds.json' % args.dataset_name, 'w+') as f:
	json.dump([c._asdict() for c in all_upds], f, indent=4, cls=NumpyEncoder)
