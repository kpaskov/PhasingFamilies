import os
import json
from collections import defaultdict
import numpy as np
import argparse
import cvxpy as cp
import scipy.sparse as sparse
import random

parser = argparse.ArgumentParser(description='Estimate recombination rate.')
parser.add_argument('assembly', type=str, help='Reference genome assembly for data.')
parser.add_argument('out_file', type=str, help='Output filename.')
parser.add_argument('kfold_crossvalidation', type=int, help='Number of fold to be used for cross validation.')
parser.add_argument('lamb', type=float, help='Regularization weight.')
parser.add_argument('phase_dirs', type=str, nargs='+', help='Directories with phase information.')
parser.add_argument('--kong_crossovers_bed_mat', type=str, default=None, help='Bed file with maternal crossovers.')
parser.add_argument('--kong_crossovers_bed_pat', type=str, default=None, help='Bed file with paternal crossovers.')

args = parser.parse_args()

# From GRCh37.p13 https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh37.p13
chrom_lengths37 = {
	'1': 249250621,
	'2': 243199373,
	'3': 198022430,
	'4': 191154276,
	'5': 180915260,
	'6': 171115067,
	'7': 159138663,
	'8': 146364022,
	'9': 141213431,
	'10': 135534747,
	'11': 135006516,
	'12': 133851895,
	'13': 115169878,
	'14': 107349540,
	'15': 102531392,
	'16': 90354753,
	'17': 81195210,
	'18': 78077248,
	'19': 59128983,
	'20': 63025520,
	'21': 48129895,
	'22': 51304566,
	'X': 155270560,
	'Y': 59373566
}

# From GRCh38.p13 https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh38.p13
chrom_lengths38 = {
	'1': 248956422,
	'2': 242193529,
	'3': 198295559,
	'4': 190214555,
	'5': 181538259,
	'6': 170805979,
	'7': 159345973,
	'8': 145138636,
	'9': 138394717,
	'10': 133797422,
	'11': 135086622,
	'12': 133275309,
	'13': 114364328,
	'14': 107043718,
	'15': 101991189,
	'16': 90338345,
	'17': 83257441,
	'18': 80373285,
	'19': 58617616,
	'20': 64444167,
	'21': 46709983,
	'22': 50818468,
	'X': 156040895,
	'Y': 57227415
}

if args.assembly == '38':
    chrom_lengths = chrom_lengths38
else:
    chrom_lengths = chrom_lengths37

# pull crossover information
quad_to_crossovers = defaultdict(list)
already_included = set()
for phase_dir in args.phase_dirs:
	print('loading phase for', phase_dir)
	for file in sorted(os.listdir(phase_dir)):
		if file.endswith('.crossovers.json'):
			with open('%s/%s' % (phase_dir, file), 'r') as f:
				cos = json.load(f)
				if len(cos)>=250:
					#print('Are they related?', cos[0]['child'][0], cos[0]['child'][1], file)
					pass
				elif cos[0]['child'][0] in already_included or cos[0]['child'][1] in already_included:
					#print('Child(ren) already included in analysis', cos[0]['child'][0], cos[0]['child'][1], file)
					pass
				else:
					for co in cos:
						quad_to_crossovers[tuple(co['child'])].append(co)
					already_included.add(co['child'][0])
					already_included.add(co['child'][1])
            
print('children with phase information', 2*len(quad_to_crossovers))

# split quads into train/test
quads = sorted(quad_to_crossovers.keys())
random.shuffle(quads)
batch_size = int(np.floor(len(quads)/args.kfold_crossvalidation))
print('batch_size', batch_size)

# pull intervals
all_positions_mat, all_positions_pat = set(), set()
num_crossovers_mat, num_crossovers_pat = 0, 0
chroms = set()
for quad, crossovers in quad_to_crossovers.items():
	for co in crossovers:
		if co['is_mat']:
			all_positions_mat.add((co['chrom'], co['start_pos']))
			all_positions_mat.add((co['chrom'], co['end_pos']))
			chroms.add(co['chrom'])
			num_crossovers_mat += 1
		if co['is_pat']:
			all_positions_pat.add((co['chrom'], co['start_pos']))
			all_positions_pat.add((co['chrom'], co['end_pos']))
			chroms.add(co['chrom'])
			num_crossovers_pat += 1

chrom_to_min, chrom_to_max = dict([(chrom, chrom_lengths[str(chrom)]) for chrom in chroms]), dict([(chrom, 1) for chrom in chroms])
kong_crossovers_mat = []
if args.kong_crossovers_bed_mat is not None:
	with open(args.kong_crossovers_bed_mat, 'r') as f:
		for line in f:
			chrom, pos = line.strip().split(':')
			start_pos, end_pos = map(int, pos.split('-'))
			if chrom[3:].isnumeric():
				chrom = int(chrom[3:])
				all_positions_mat.add((chrom, start_pos))
				all_positions_mat.add((chrom, end_pos))
				num_crossovers_mat += 1
				chrom_to_min[chrom] = min(chrom_to_min[chrom], start_pos)
				chrom_to_max[chrom] = max(chrom_to_max[chrom], end_pos)
				kong_crossovers_mat.append({'chrom': chrom, 'start_pos': start_pos, 'end_pos': end_pos})
random.shuffle(kong_crossovers_mat)

kong_crossovers_pat = []
if args.kong_crossovers_bed_pat is not None:
	with open(args.kong_crossovers_bed_pat, 'r') as f:
		for line in f:
			chrom, pos = line.strip().split(':')
			start_pos, end_pos = map(int, pos.split('-'))
			if chrom[3:].isnumeric():
				chrom = int(chrom[3:])
				all_positions_pat.add((chrom, start_pos))
				all_positions_pat.add((chrom, end_pos))
				num_crossovers_pat += 1
				chrom_to_min[chrom] = min(chrom_to_min[chrom], start_pos)
				chrom_to_max[chrom] = max(chrom_to_max[chrom], end_pos)
				kong_crossovers_pat.append({'chrom': chrom, 'start_pos': start_pos, 'end_pos': end_pos})
random.shuffle(kong_crossovers_pat)

#print(chrom_to_min, chrom_to_max)

batch_size_kong_mat = int(np.floor(len(kong_crossovers_mat)/args.kfold_crossvalidation))
batch_size_kong_pat = int(np.floor(len(kong_crossovers_pat)/args.kfold_crossvalidation))

all_positions = sorted(all_positions_mat | all_positions_pat | set([(chrom, 1) for chrom in chroms]) | set([(chrom, chrom_lengths[str(chrom)]) for chrom in chroms]))
pos_to_index = dict([(x, i) for i, x in enumerate(all_positions)])
lengths = np.array([0 if all_positions[i][0] != all_positions[i+1][0] else all_positions[i+1][1]-all_positions[i][1] for i in range(len(all_positions)-1)])
print('num_crossovers', num_crossovers_mat+num_crossovers_pat, 'positions', len(all_positions))

kong_region = np.ones((len(all_positions),), dtype=bool)
for chrom in chroms:
	chrom_min, chrom_max = chrom_to_min[chrom], chrom_to_max[chrom]
	kong_region[[c==chrom and p<chrom_min for c, p in all_positions]] = False
	kong_region[[c==chrom and p>chrom_max for c, p in all_positions]] = False
print('genome in kong', np.sum(kong_region[:-1]*lengths)/np.sum(lengths))

np.save('%s.lengths' % (args.out_file), lengths)
with open('%s.positions.txt' % (args.out_file), 'w+') as f:
	for p in all_positions:
		f.write('%d\t%d\n' % p)

# create feature matrix
def X_from_crossovers(crossovers, pos_to_index):
	data, row_ind, col_ind = [], [], []
	for row_i, co in enumerate(crossovers):
		start_index, end_index = pos_to_index[(co['chrom'], co['start_pos'])], pos_to_index[(co['chrom'], co['end_pos'])]
		data.extend([1,]*(end_index-start_index))
		row_ind.extend([row_i,]*(end_index-start_index))
		col_ind.extend(range(start_index, end_index))
	return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(crossovers), len(pos_to_index)))

# estimate recombination rates
def estimate_recombination_rates(X, lengths, positions, kong_region, num_kong):
	m, n = X.shape
	print('Estimating...', m, n)

	p = cp.Variable(n)

	# maximize likelihood of multinomial loss...
	expr = cp.sum(cp.log(X@p))/m

	if num_kong>0:
		expr -= num_kong*cp.log(cp.sum(p[kong_region]))/m

	# and regularize the difference between neighboring intervals (within each chromosome)
	current_chrom = positions[0][0]
	current_chrom_start = 0
	for i in range(1, len(positions)):
		if positions[i][0] != current_chrom:
			expr -= cp.tv(args.lamb*p[current_chrom_start:i]/lengths[current_chrom_start:i])
			current_chrom, current_chrom_start = positions[i][0], i
	expr -= cp.tv(args.lamb*p[current_chrom_start:i]/lengths[current_chrom_start:i])

	# now solve
	prob = cp.Problem(cp.Maximize(expr), [p >= 0, p <= 1, cp.sum(p)==1])
	result = prob.solve(solver='MOSEK', mosek_params={'MSK_IPAR_INTPNT_MAX_ITERATIONS': 300}, verbose=True)
	#result = prob.solve(solver='ECOS', max_iters=200, verbose=True)
	print(prob.status)

	ps = np.clip([v for v in p.value], 0, 1) # clip just in case we have some numerical problems
	if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
		raise Error('Parameters not fully estimated.')
	return ps

for batch_num in range(args.kfold_crossvalidation):
	test_quads = set(quads[(batch_size*batch_num):(batch_size*(batch_num+1))])
	train_quads = set(quads)-test_quads

	with open('%s.train_quads.%g.%d.txt' % (args.out_file, args.lamb, batch_num), 'w+') as f:
		for quad in train_quads:
			f.write('\t'.join(quad) + '\n')

	with open('%s.test_quads.%g.%d.txt' % (args.out_file, args.lamb, batch_num), 'w+') as f:
		for quad in test_quads:
			f.write('\t'.join(quad) + '\n')

	crossovers_mat_train = sum([[co for co in quad_to_crossovers[quad] if co['is_mat']] for quad in train_quads], [])
	crossovers_mat_test = sum([[co for co in quad_to_crossovers[quad] if co['is_mat']] for quad in test_quads], [])
	crossovers_pat_train = sum([[co for co in quad_to_crossovers[quad] if co['is_pat']] for quad in train_quads], [])
	crossovers_pat_test = sum([[co for co in quad_to_crossovers[quad] if co['is_pat']] for quad in test_quads], [])

	crossovers_mat_train_kong = kong_crossovers_mat[0:(batch_size_kong_mat*batch_num)] + kong_crossovers_mat[(batch_size_kong_mat*(batch_num+1)):]
	crossovers_pat_train_kong = kong_crossovers_pat[0:(batch_size_kong_pat*batch_num)] + kong_crossovers_pat[(batch_size_kong_pat*(batch_num+1)):]
	crossovers_mat_test_kong = kong_crossovers_mat[(batch_size_kong_mat*batch_num):(batch_size_kong_mat*(batch_num+1))]
	crossovers_pat_test_kong = kong_crossovers_pat[(batch_size_kong_pat*batch_num):(batch_size_kong_pat*(batch_num+1))]

	X_mat_train = X_from_crossovers(crossovers_mat_train+crossovers_mat_train_kong, pos_to_index)
	X_mat_test = X_from_crossovers(crossovers_mat_test+crossovers_mat_test_kong, pos_to_index)
	X_pat_train = X_from_crossovers(crossovers_pat_train+crossovers_pat_train_kong, pos_to_index)
	X_pat_test = X_from_crossovers(crossovers_pat_test+crossovers_pat_test_kong, pos_to_index)

	#sparse.save_npz('%s.X_mat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_train)
	#sparse.save_npz('%s.X_mat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_test)
	#sparse.save_npz('%s.X_pat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_train)
	#sparse.save_npz('%s.X_pat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_test)

	ps_mat = estimate_recombination_rates(X_mat_train, lengths, all_positions, kong_region, len(crossovers_mat_train_kong))
	np.save('%s.ps_mat.%g.%d' % (args.out_file, args.lamb, batch_num), ps_mat)
	np.save('%s.crossover_ps_mat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_train.dot(ps_mat))
	np.save('%s.crossover_ps_mat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_test.dot(ps_mat))
	np.save('%s.crossover_lengths_mat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_train.dot(lengths))
	np.save('%s.crossover_lengths_mat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_test.dot(lengths))

	ps_pat = estimate_recombination_rates(X_pat_train, lengths, all_positions, kong_region, len(crossovers_pat_train_kong))
	np.save('%s.ps_pat.%g.%d' % (args.out_file, args.lamb, batch_num), ps_pat)
	np.save('%s.crossover_ps_pat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_train.dot(ps_pat))
	np.save('%s.crossover_ps_pat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_test.dot(ps_pat))
	np.save('%s.crossover_lengths_pat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_train.dot(lengths))
	np.save('%s.crossover_lengths_pat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_test.dot(lengths))

           