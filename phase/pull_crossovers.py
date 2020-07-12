import numpy as np
from collections import namedtuple, defaultdict, Counter
from itertools import product, combinations
import json
import argparse
import input_output
import os

parser = argparse.ArgumentParser(description='Pull crossovers from phasing output.')
parser.add_argument('phase_dir', type=str, help='Directory with phase data.')
parser.add_argument('ped_file', type=str, help='Ped file for data.')
parser.add_argument('assembly', type=str, help='Reference genome assembly for data.')

args = parser.parse_args()

# start by pulling families, only consider nuclear families
families = input_output.pull_families(args.ped_file)
families = [x for x in families if x.num_ancestors()==2 and len(x.ordered_couples)==1]

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
			chrs.append(int(pieces[0][3:]))
			states.append(list(map(int, pieces[1:-2])))
			starts.append(int(pieces[-2]))
			ends.append(int(pieces[-1]))

		mat_indices = [i for i, x in enumerate(header) if x.endswith('_mat')]
		pat_indices = [i for i, x in enumerate(header) if x.endswith('_pat')]

		states = np.array(states).T
		starts = np.array(starts)
		ends = np.array(ends)
		for (mat_index1, pat_index1), (mat_index2, pat_index2) in combinations(zip(mat_indices[2:], pat_indices[2:]), 2):
			if np.sum((states[mat_index1, :] == states[mat_index2, :])*(ends-starts))/np.sum(ends-starts) > 0.9 and \
			   np.sum((states[pat_index1, :] == states[pat_index2, :])*(ends-starts))/np.sum(ends-starts) > 0.9:
			   raise Exception('This family contains identical twins.')


	return states, np.array(chrs), starts, ends, individuals, mat_indices, pat_indices
		

# extracts all recombination points from phase
Recombination = namedtuple('Recombination', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'family_size'])
def pull_recombinations(family_id, states, chroms, starts, ends, individuals,  indices, is_mat):
    recombinations = []
    for chrom in range(1, 23):
        # block out 
        
        for individual, index in zip(individuals, indices):

            change_indices = np.where((chroms[:-1]==chrom) & (chroms[1:]==chrom) & (states[index, :-1] != states[index, 1:]))[0]+1
                        
            if change_indices.shape[0]>0:
                current_index = change_indices[0]
                for next_index in change_indices[1:]:
                    assert states[index, current_index-1] != states[index, current_index]
                    assert states[index, current_index] != states[index, next_index]

                    if states[index, current_index-1] != -1 and states[index, current_index] != -1:
                        # we know exactly where the recombination happened
                        recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[current_index]),
                                    (individual, individuals[2]), is_mat, not is_mat, len(individuals)))
                    elif states[index, current_index-1] != -1 and states[index, current_index] == -1 and states[index, next_index] != -1 and states[index, current_index-1] != states[index, next_index]:
                        # there's a region where the recombination must have occured
                        recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[next_index]),
                                    (individual, individuals[2]), is_mat, not is_mat, len(individuals)))
                    
                    current_index = next_index
                if states[index, current_index-1] != -1 and states[index, current_index] != -1:
                    # we know exactly where the recombination happened
                    recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[current_index]),
                                    (individual, individuals[2]), is_mat, not is_mat, len(individuals)))

    return recombinations

Crossover = namedtuple('Crossover', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'recombinations', 'family_size'])
GeneConversion = namedtuple('GeneConversion', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'recombinations', 'family_size'])

def match_recombinations(recombinations, chrom, child, is_mat):
    gene_conversions = []
    crossovers = []
    
    rs = [r for r in recombinations if r.chrom==chrom and r.child==child and r.is_mat==is_mat]
    dists = np.array([r2.start_pos-r1.end_pos for r1, r2 in zip(rs[:-1], rs[1:])])
    breaks = [0] + (np.where(dists>500000)[0]+1).tolist() + [len(rs)]
    for break_start, break_end in zip(breaks[:-1], breaks[1:]):
        r_group = rs[break_start:break_end]
        if len(r_group)>0:
	        if len(r_group)%2 == 0:
	            gene_conversions.append(GeneConversion(family_id, chrom, 
	                                                   r_group[0].start_pos, r_group[-1].end_pos, child,
	                                                   is_mat, not is_mat, len(r_group)>2, tuple(r_group), r_group[0].family_size))
	        else:
	            crossovers.append(Crossover(family_id, chrom,
	                                       r_group[0].start_pos, r_group[-1].end_pos, child,
	                                       is_mat, not is_mat, len(r_group)>1, tuple(r_group), r_group[0].family_size))
                    
    return gene_conversions, crossovers

def remove_massive_events(gene_conversions, crossovers):
    # remove events that span too large an area
    gene_conversions = [gc for gc in gene_conversions if gc.end_pos-gc.start_pos<2000000]

    # don't remove any crossovers because it will affect the phasing
    #crossovers = [co for co in crossovers if co.end_pos-co.start_pos<2000000]

    return gene_conversions, crossovers
    

def move_identical(gene_conversions, crossovers):
    # gene conversions
    gc_to_overlapping = dict([(x, {x}) for x in gene_conversions])
    for gc1, gc2 in combinations(gene_conversions, 2):
        if gc1.child != gc2.child and gc1.chrom == gc2.chrom and min(gc1.end_pos, gc2.end_pos)-max(gc1.start_pos, gc2.start_pos)>0: 
            gc_to_overlapping[gc1].add(gc2)
            gc_to_overlapping[gc2].add(gc1)

    to_be_added = []
    to_be_removed = set()
    for key, gc_group in gc_to_overlapping.items():
        if set(sum([list(x.child) for x in gc_group], [])) == set(individuals[2:]):
            to_be_added.append(GeneConversion(key.family, key.chrom, 
                                              min([gc.start_pos for gc in gc_group]), max([gc.end_pos for gc in gc_group]),
                                              individuals[2], key.is_mat, key.is_pat, key.is_complex, 
                                              tuple(sum([list(gc.recombinations) for gc in gc_group], [])), key.family_size))
            to_be_removed.update(gc_group)


    for c in to_be_removed:
        gene_conversions.remove(c)
    gene_conversions = [GeneConversion(gc.family, gc.chrom, gc.start_pos, gc.end_pos,
                                      gc.child[0], gc.is_mat, gc.is_pat, gc.is_complex, gc.recombinations, gc.family_size) for gc in gene_conversions]
    gene_conversions.extend(to_be_added)
    print('gene conversions transferred to child1', len(to_be_added))

    # recombinations
    co_to_overlapping = dict([(x, {x}) for x in crossovers])
    for gc1, gc2 in combinations(crossovers, 2):
        if gc1.child != gc2.child and gc1.chrom == gc2.chrom and min(gc1.end_pos, gc2.end_pos)-max(gc1.start_pos, gc2.start_pos)>0: 
            co_to_overlapping[gc1].add(gc2)
            co_to_overlapping[gc2].add(gc1)

    to_be_added = []
    to_be_removed = set()
    for key, gc_group in co_to_overlapping.items():
        if set(sum([list(x.child) for x in gc_group], [])) == set(individuals[2:]):
            to_be_added.append(Crossover(key.family, key.chrom, 
                                              min([gc.start_pos for gc in gc_group]), max([gc.end_pos for gc in gc_group]),
                                              individuals[2], key.is_mat, key.is_pat, key.is_complex, 
                                              tuple(sum([list(gc.recombinations) for gc in gc_group], [])), key.family_size))
            to_be_removed.update(gc_group)


    for c in to_be_removed:
        crossovers.remove(c)
    crossovers = [Crossover(co.family, co.chrom, co.start_pos, co.end_pos,
                                      co.child[0], co.is_mat, co.is_pat, co.is_complex, co.recombinations, co.family_size) for co in crossovers]
    crossovers.extend(to_be_added)
    print('crossovers transferred to child1', len(to_be_added))
    return gene_conversions, crossovers
	
for file in sorted(os.listdir(args.phase_dir)):
	if file.endswith('.phased.txt'):
		try:
			family_id = file[:-11]
			print(family_id)
			states, chroms, starts, ends, individuals, mat_indices, pat_indices = pull_phase('%s/%s.phased.txt' % (args.phase_dir, family_id))
			mult = ends-starts
			num_children = len(individuals)-2

			# start by pulling all recombinations
			mat_recombinations = pull_recombinations(family_id, states, chroms, starts, ends, individuals, mat_indices, True)
			pat_recombinations = pull_recombinations(family_id, states, chroms, starts, ends, individuals, pat_indices, False)
			recombinations = mat_recombinations + pat_recombinations

			# now identify crossovers and gene conversions
			gene_conversions, crossovers = [], []
			children = set([x.child for x in recombinations])
			for chrom in range(1, 23):
				for child in children:
					gc, co = match_recombinations(recombinations, chrom, child, True)
					gene_conversions.extend(gc)
					crossovers.extend(co)

					gc, co = match_recombinations(recombinations, chrom, child, False)
					gene_conversions.extend(gc)
					crossovers.extend(co)

			gene_conversions, crossovers = remove_massive_events(gene_conversions, crossovers)

			#if num_children>2:
			#	gene_conversions, crossovers = move_identical(gene_conversions, crossovers)

			print('gc', len(gene_conversions), 'co', len(crossovers))

			gc = len(gene_conversions)/num_children
			cr = len(crossovers)/num_children
			print('avg gene conversion', gc, 'avg crossover', cr)

			with open('%s/%s.crossovers.json' % (args.phase_dir, family_id), 'w+') as f:
				json.dump([c._asdict() for c in crossovers], f, indent=4)

			with open('%s/%s.gene_conversions.json' % (args.phase_dir, family_id), 'w+') as f:
				json.dump([gc._asdict() for gc in gene_conversions], f, indent=4)
		except Exception as e: print(e)


