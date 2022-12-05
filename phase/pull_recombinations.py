import numpy as np
from collections import namedtuple, defaultdict, Counter
from itertools import product, combinations
import json
import argparse
import input_output
import os
import traceback
from numpyencoder import NumpyEncoder
import traceback

parser = argparse.ArgumentParser(description='Pull crossovers from phasing output.')
parser.add_argument('dataset_name', type=str, help='Name of dataset.')

chroms_of_interest = [str(x) for x in range(1, 23)] + ['X']
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
			if len(pieces) != len(header)+3:
				print('ERROR', pieces)
			chrs.append(pieces[0][3:])
			states.append(list(map(int, pieces[1:-2])))
			starts.append(int(pieces[-2]))
			ends.append(int(pieces[-1]))

		mat_indices = [i for i, x in enumerate(header) if x.endswith('_mat')]
		pat_indices = [i for i, x in enumerate(header) if x.endswith('_pat')]

		states = np.array(states).T
		starts = np.array(starts)
		ends = np.array(ends)

		if states.shape == (0,):
			raise Exception('No data')

		# if this is a hard to sequences region, we don't know the exact location of crossovers
		states[:-1, (states[-1, :]==1)] = -1

		# if there's UPD, it's not a crossover
		for mat_index, pat_index in zip(mat_indices[2:], pat_indices[2:]):
			states[:, states[mat_index, :]==2] = -1
			states[:, states[mat_index, :]==3] = -1
			states[:, states[pat_index, :]==0] = -1
			states[:, states[pat_index, :]==1] = -1

	return states, chrs, starts, ends, individuals, mat_indices, pat_indices
		

# extracts deletions from within recombination interval
def extract_deletions(states, starts, ends, is_mat):
	#print(states)
	if is_mat:
		del_indices = [0, 1]
	else:
		del_indices = [2, 3]

	return np.any(states[del_indices, :-1]==0)


# extracts all recombination points from phase
Recombination = namedtuple('Recombination', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'family_size', 'mom', 'dad', 
	'is_left_match', 'is_right_match', 'is_other_parent_match', 'is_other_parent_mismatch', 'is_hts'])
def pull_recombinations(family_id, states, chroms, starts, ends, individuals,  indices, other_indices, is_mat):
    match_state = 0 if is_mat else 2
    other_match_state = 2 if is_mat else 0

    def make_recombination(chrom, start_index, end_index, individual, index, other_index):
    	return Recombination(family_id, chrom, int(ends[start_index])-1, int(starts[end_index]),
                                    (individual, individuals[2]), is_mat, not is_mat, 
                                    len(individuals), individuals[0], individuals[1],
                                    states[index, start_index]==match_state, 
                                    states[index, end_index]==match_state,
                                    np.all(states[other_index, start_index:end_index+1]==other_match_state),
                                    np.all(states[other_index, start_index:end_index+1]!=-1) and np.all(states[other_index, start_index:end_index+1]!=other_match_state),
                                    np.any(states[-1, start_index:end_index+1]==1))

    recombinations = []
    for chrom in chroms_of_interest:
        # block out 
        is_chrom = np.array([x==chrom for x in chroms])
        for individual, index, other_index in zip(individuals, indices, other_indices):
            change_indices = np.where(is_chrom[:-1] & is_chrom[1:] & (states[index, :-1] != states[index, 1:]))[0]+1

            if change_indices.shape[0]>0:
                current_index = change_indices[0]
                for next_index in change_indices[1:]:
                    assert states[index, current_index-1] != states[index, current_index]
                    assert states[index, current_index] != states[index, next_index]

                    if states[index, current_index-1] != -1 and states[index, current_index] != -1:
                        # we know exactly where the recombination happened
                        recombinations.append(make_recombination(chrom, current_index-1, current_index, individual, index, other_index))

                    elif states[index, current_index-1] != -1 and states[index, current_index] == -1 and states[index, current_index-1] != states[index, next_index]:
                        # there's a region where the recombination must have occured
                        recombinations.append(make_recombination(chrom, current_index-1, next_index, individual, index, other_index))
                    current_index = next_index
                
                if states[index, current_index-1] != -1 and states[index, current_index] != -1:
                    # we know exactly where the recombination happened
                    recombinations.append(make_recombination(chrom, current_index-1, current_index, individual, index, other_index))
    return recombinations

Crossover = namedtuple('Crossover', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'is_hts', 'recombinations', 'family_size', 'mom', 'dad'])
GeneConversion = namedtuple('GeneConversion', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'is_hts', 'recombinations', 'family_size', 'mom', 'dad'])

def match_recombinations(recombinations, chrom, family_id, child, is_mat):
    gene_conversions = []
    crossovers = []
    
    rs = [r for r in recombinations if r.chrom==chrom and r.child==child and r.is_mat==is_mat]
    dists = np.array([r2.start_pos-r1.end_pos for r1, r2 in zip(rs[:-1], rs[1:])])
    breaks = [0] + (np.where(dists>100000)[0]+1).tolist() + [len(rs)]
    for break_start, break_end in zip(breaks[:-1], breaks[1:]):
        r_group = rs[break_start:break_end]
        if len(r_group)>0:
            if len(r_group)%2 == 0:
                gene_conversions.append(GeneConversion(family_id, chrom, 
                                                       r_group[0].start_pos, r_group[-1].end_pos, child,
                                                       is_mat, not is_mat, 
                                                       len(r_group)>2, # is_complex
                                                       np.any([r.is_hts for r in r_group]), # is_hts
                                                       [r._asdict() for r in r_group], r_group[0].family_size,
                                                       r_group[0].mom, r_group[0].dad))
            else:
                crossovers.append(Crossover(family_id, chrom,
                                           r_group[0].start_pos, r_group[-1].end_pos, child,
                                           is_mat, not is_mat, 
                                           len(r_group)>1, # is_complex
                                           np.any([r.is_hts for r in r_group]), # is_hts
                                           [r._asdict() for r in r_group], r_group[0].family_size,
                                           r_group[0].mom, r_group[0].dad))
                    
    return gene_conversions, crossovers

try:
	with open('%s/sibpairs.json' % args.dataset_name, 'r') as f:
		sibpairs = json.load(f)
except:
	with open('%s/similarity.txt' % args.dataset_name, 'r') as f:
		sibpairs = []
		next(f) # skip header
		for line in f:
			sibpairs.append({'phase_dir': args.dataset_name, 'family': line.split('\t', maxsplit=1)[0]})

all_recombinations = []
all_crossovers = []
all_gene_conversions = []
for sibpair in sibpairs:
	family_key = sibpair['family']
	print(family_key)
	states, chroms, starts, ends, individuals, mat_indices, pat_indices = pull_phase('%s/%s.phased.txt' % (sibpair['phase_dir'], family_key))
	
	missing_chroms = set(chroms_of_interest) - set(np.unique(chroms))
	if len(missing_chroms)>0:
		#raise Exception('missing %s' % str(missing_chroms), family_key)
		print('missing %s' % str(missing_chroms), family_key)

	mult = ends-starts
	num_children = len(individuals)-2

	# start by pulling all recombinations
	mat_recombinations = pull_recombinations(family_key, states, chroms, starts, ends, individuals, mat_indices, pat_indices, True)
	pat_recombinations = pull_recombinations(family_key, states, chroms, starts, ends, individuals, pat_indices, mat_indices, False)
	recombinations = mat_recombinations + pat_recombinations

	all_recombinations.extend(recombinations)

with open('%s/recombinations.json' % args.dataset_name, 'w+') as f:
	json.dump([c._asdict() for c in all_recombinations], f, indent=4, cls=NumpyEncoder)	


