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


has_missing_chroms = []
has_missing_individuals = []
is_missing_phase_data = []
all_good = 0

for family in families:
    if os.path.isfile('%s/%s.phased.txt' % (args.phase_dir, family.id)):
        chroms, individuals = pull_phased_chroms('%s/%s.phased.txt' % (args.phase_dir, family.id))
        missing_chroms = set([str(x) for x in range(1, 23)]) - chroms
        if len(missing_chroms)>0:
            has_missing_chroms.append(family.id)
        elif len(individuals) != len(family.individuals):
            has_missing_individuals.append((family.id, set(family.individuals)-set(individuals)))
        else:
            all_good += 1
    else:
        is_missing_phase_data.append(family.id)

print('has missing chroms', has_missing_chroms)
print('has missing individuals', has_missing_individuals)
print('is missing phase data', is_missing_phase_data)
print('all good', all_good)


