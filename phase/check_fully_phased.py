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
parser.add_argument('data_dir', type=str, help='Directory with genotype data.')
parser.add_argument('param_file', type=str, help='Parameter file.')

args = parser.parse_args()

# start by pulling families, only consider nuclear families
families = input_output.pull_families(args.ped_file)
families = [x for x in families if x.num_ancestors()==2 and len(x.ordered_couples)==1]

sample_file = '%s/samples.json' % args.data_dir
with open(sample_file, 'r') as f:
    sample_ids = set(json.load(f))

with open(args.param_file, 'r') as f:
    params = json.load(f)

for family in families:
    to_be_removed = [x for x in family.individuals if x not in sample_ids]
    family.prune(to_be_removed)

families = [x for x in families if x.num_descendents()>0]
print(len(families), 'have genomic data')

for family in families:
    to_be_removed = [x for x in family.individuals if (x not in params and '%s.%s' % (family.id, x) not in params)]
    family.prune(to_be_removed)

families = [x for x in families if x.num_descendents()>0]
print(len(families), 'have parameters')

#pulls phase data from a file
def pull_phased_chroms(filename):
	with open(filename, 'r') as f:
		header = next(f).strip().split('\t')[1:-2] # skip header
		individuals = [x[:-4] for x in header if x.endswith('_mat')]

		chrs = set()
		for line in f:
			pieces = line.strip().split('\t', maxsplit=1)
			chrs.add(pieces[0][3:])
			
	return chrs, individuals

def write_to_file(outf, chrom, positions, states):
	# write final states to file
	change_indices = [-1] + np.where(np.any(states[1:, :]!=states[:-1, :], axis=1))[0].tolist() + [positions.shape[0]-1]
	for j in range(1, len(change_indices)):
		start_index, end_index = change_indices[j-1]+1, change_indices[j]
		outf.write('%s\t%s\t%d\t%d\n' % (
						'chr' + chrom, 
						'\t'.join(map(str, states[start_index, :])), 
						positions[start_index, 0], positions[end_index, 1]))

# writes a recomb only file
def create_recomb_file(filename):
	states = []
	positions = []
	current_chrom = None
	with open(filename, 'r') as f, open(filename + '.recomb.txt', 'w+') as outf:
		header = next(f).strip().split('\t')[1:-2] # skip header

		for line in f:
			pieces = line.strip().split('\t')
			chrom = pieces[0][3:]
			if current_chrom is None: 
				current_chrom = chrom
				positions.append([int(x) for x in pieces[-2:]])
				states.append([int(x) for x in pieces[1:-2]])
			elif chrom == current_chrom:
				positions.append([int(x) for x in pieces[-2:]])
				states.append([int(x) for x in pieces[1:-2]])
			else:
				# write data from previous chrom
				states = np.array(states)
				positions = np.array(positions)
				
				# ignore deletions and haplotypes, we only care about crossovers
				states[:, :4] = -1

				# if this is a hard to sequences region, we don't know the exact location of crossovers
				states[states[:, -1]==1, :] = -1

				# ignore hard to sequence regions as well
				#states[:, -1] = -1

				write_to_file(outf, current_chrom, positions, states)

				# reset for next chrom
				current_chrom = chrom
				positions = [[int(x) for x in pieces[-2:]]]
				states = [[int(x) for x in pieces[1:-2]]]

		# write data from last chrom
		states = np.array(states)
		positions = np.array(positions)
					
		# ignore deletions and haplotypes, we only care about crossovers
		states[:, :4] = -1

		# if this is a hard to sequences region, we don't know the exact location of crossovers
		states[states[:, -1]==1, :] = -1

		# ignore hard to sequence regions as well
		#states[:, -1] = -1

		write_to_file(outf, current_chrom, positions, states)
			
has_no_data = []
has_missing_chroms = []
has_missing_individuals = []
is_missing_phase_data = []
all_good = 0

for family in families:
	phase_filename = '%s/%s.phased.txt' % (args.phase_dir, family.id)
	if os.path.isfile(phase_filename):
		try:
			chroms, individuals = pull_phased_chroms(phase_filename)
			missing_chroms = set([str(x) for x in range(1, 23)]) - chroms
			if len(missing_chroms)>0:
				has_missing_chroms.append(family.id)
			elif len(individuals) != len(family.individuals):
				has_missing_individuals.append((family.id, set(family.individuals)-set(individuals)))
			else:
				all_good += 1
				create_recomb_file(phase_filename)
		except StopIteration:
			has_no_data.append(family.id)
	else:
		is_missing_phase_data.append(family.id)

print('has no data', has_no_data)
print('has missing chroms', has_missing_chroms)
print('has missing individuals', has_missing_individuals)
print('is missing phase data', is_missing_phase_data)
print('all good', all_good)


