import numpy as np
from collections import namedtuple, defaultdict, Counter
from itertools import product, combinations
import json
import argparse
import input_output
import os
import traceback

parser = argparse.ArgumentParser(description='Pull crossovers from phasing output.')
parser.add_argument('phase_dir', type=str, help='Directory with phase data.')
parser.add_argument('ped_file', type=str, help='Ped file for data.')
parser.add_argument('assembly', type=str, help='Reference genome assembly for data.')

chroms_of_interest = [str(x) for x in range(1, 23)] + ['X']
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

		for (mat_index1, pat_index1), (mat_index2, pat_index2) in combinations(zip(mat_indices[2:], pat_indices[2:]), 2):
			no_missing_mat = (states[mat_index1, :] != -1) & (states[mat_index2, :] != -1)
			no_missing_pat = (states[pat_index1, :] != -1) & (states[pat_index2, :] != -1)

			if np.sum(((states[mat_index1, :] == states[mat_index2, :])*(ends-starts))[no_missing_mat])/np.sum((ends-starts)[no_missing_mat]) > 0.9 and \
			   np.sum(((states[pat_index1, :] == states[pat_index2, :])*(ends-starts))[no_missing_pat])/np.sum((ends-starts)[no_missing_pat]) > 0.9:
			   raise Exception('This family contains identical twins.')


	return states, chrs, starts, ends, individuals, mat_indices, pat_indices
		

# extracts all recombination points from phase
Recombination = namedtuple('Recombination', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'family_size', 'mom', 'dad'])
def pull_recombinations(family_id, states, chroms, starts, ends, individuals,  indices, is_mat):
    recombinations = []
    for chrom in chroms_of_interest:
        # block out 
        is_chrom = np.array([x==chrom for x in chroms])
        for individual, index in zip(individuals, indices):
            change_indices = np.where(is_chrom[:-1] & is_chrom[1:] & (states[index, :-1] != states[index, 1:]))[0]+1
                        
            if change_indices.shape[0]>0:
                current_index = change_indices[0]
                for next_index in change_indices[1:]:
                    assert states[index, current_index-1] != states[index, current_index]
                    assert states[index, current_index] != states[index, next_index]

                    if states[index, current_index-1] != -1 and states[index, current_index] != -1:
                        # we know exactly where the recombination happened
                        recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[current_index]),
                                    (individual, individuals[2]), is_mat, not is_mat, len(individuals), individuals[0], individuals[1]))
                    elif states[index, current_index-1] != -1 and states[index, current_index] == -1 and states[index, next_index] != -1 and states[index, current_index-1] != states[index, next_index]:
                        # there's a region where the recombination must have occured
                        recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[next_index]),
                                    (individual, individuals[2]), is_mat, not is_mat, len(individuals), individuals[0], individuals[1]))
                    
                    current_index = next_index
                if states[index, current_index-1] != -1 and states[index, current_index] != -1:
                    # we know exactly where the recombination happened
                    recombinations.append(Recombination(family_id, chrom, int(ends[current_index-1])-1, int(starts[current_index]),
                                    (individual, individuals[2]), is_mat, not is_mat, len(individuals), individuals[0], individuals[1]))

    return recombinations

Crossover = namedtuple('Crossover', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'recombinations', 'family_size', 'mom', 'dad'])
GeneConversion = namedtuple('GeneConversion', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat', 'is_complex', 'recombinations', 'family_size', 'mom', 'dad'])

def match_recombinations(recombinations, chrom, child, is_mat):
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
                                                       is_mat, not is_mat, len(r_group)>2, tuple(r_group), r_group[0].family_size,
                                                       r_group[0].mom, r_group[0].dad))
            else:
                crossovers.append(Crossover(family_id, chrom,
                                           r_group[0].start_pos, r_group[-1].end_pos, child,
                                           is_mat, not is_mat, len(r_group)>1, tuple(r_group), r_group[0].family_size,
                                           r_group[0].mom, r_group[0].dad))
                    
    return gene_conversions, crossovers
	
for file in sorted(os.listdir(args.phase_dir)):
	if file.endswith('.phased.txt'):
		try:
			family_id = file[:-11]
			print(family_id)
			states, chroms, starts, ends, individuals, mat_indices, pat_indices = pull_phase('%s/%s.phased.txt' % (args.phase_dir, family_id))
			
			missing_chroms = set(chroms_of_interest) - set(np.unique(chroms))
			if len(missing_chroms)>0:
				raise Exception('missing %s' % str(missing_chroms))

			mult = ends-starts
			num_children = len(individuals)-2

			# start by pulling all recombinations
			mat_recombinations = pull_recombinations(family_id, states, chroms, starts, ends, individuals, mat_indices, True)
			pat_recombinations = pull_recombinations(family_id, states, chroms, starts, ends, individuals, pat_indices, False)
			recombinations = mat_recombinations + pat_recombinations

			# now identify crossovers and gene conversions
			gene_conversions, crossovers = [], []
			children = set([x.child for x in recombinations])
			for chrom in chroms_of_interest:
				for child in children:
					gc, co = match_recombinations(recombinations, chrom, child, True)
					gene_conversions.extend(gc)
					crossovers.extend(co)

					gc, co = match_recombinations(recombinations, chrom, child, False)
					gene_conversions.extend(gc)
					crossovers.extend(co)

			print('gc', len(gene_conversions), 'co', len(crossovers))

			gc = len(gene_conversions)/num_children
			cr = len(crossovers)/num_children
			print('avg gene conversion', gc, 'avg crossover', cr)

			with open('%s/%s.crossovers.json' % (args.phase_dir, family_id), 'w+') as f:
				json.dump([c._asdict() for c in crossovers], f, indent=4)

			with open('%s/%s.gene_conversions.json' % (args.phase_dir, family_id), 'w+') as f:
				json.dump([gc._asdict() for gc in gene_conversions], f, indent=4)
		except Exception:
			traceback.print_exc()

			crossover_file = '%s/%s.crossovers.json' % (args.phase_dir, family_id)
			if os.path.isfile(crossover_file):
				os.remove(crossover_file)
			gene_conversion_file = '%s/%s.gene_conversions.json' % (args.phase_dir, family_id)
			if os.path.isfile(gene_conversion_file):
				os.remove(gene_conversion_file)


