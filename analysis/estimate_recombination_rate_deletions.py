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
parser.add_argument('kfold_crossvalidation', type=int, help='Number of fold to be used for cross validation. 1 means no cross validation')
parser.add_argument('lamb', type=float, help='Regularization weight.')
parser.add_argument('dataset', type=str, help='Name of dataset.')

args = parser.parse_args()

chroms = [str(x) for x in range(1, 23)]

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
with open('recomb_%s/crossovers.json' % args.dataset, 'r') as f:
    crossovers = json.load(f)

with open('recomb_%s/deletions.json' % args.dataset, 'r') as f:
    deletions = json.load(f)

for co in crossovers:
	if co['start_pos'] == co['end_pos']:
		print(co)
	assert co['start_pos'] != co['end_pos']

quads = sorted(set([tuple(co['child']) for co in crossovers]))
print('children with phase information', 2*len(quads))

# split quads into train/test
random.shuffle(quads)
if args.kfold_crossvalidation == 1:
	batch_size = None
else:
	batch_size = int(np.floor(len(quads)/args.kfold_crossvalidation))
print('batch_size', batch_size)

# pull intervals
chrom_to_positions_mat, chrom_to_positions_pat = defaultdict(set), defaultdict(set)
num_crossovers_mat, num_crossovers_pat = 0, 0
num_deletions_mat, num_deletions_pat = 0, 0

for co in crossovers:
	if co['is_mat']:
		chrom_to_positions_mat[co['chrom']].add(co['start_pos'])
		chrom_to_positions_mat[co['chrom']].add(co['end_pos'])
		num_crossovers_mat += 1
		
	if co['is_pat']:
		chrom_to_positions_pat[co['chrom']].add(co['start_pos'])
		chrom_to_positions_pat[co['chrom']].add(co['end_pos'])
		num_crossovers_pat += 1

for d in deletions:
	if d['is_mat']:
		chrom_to_positions_mat[d['chrom']].add(d['start_pos'])
		chrom_to_positions_mat[d['chrom']].add(d['end_pos'])
		num_deletions_mat += 1

	if d['is_pat']:
		chrom_to_positions_pat[d['chrom']].add(d['start_pos'])
		chrom_to_positions_pat[d['chrom']].add(d['end_pos'])
		num_deletions_pat += 1

print('num_crossovers_mat', num_crossovers_mat, 'num_crossovers_pat', num_crossovers_pat)
print('num_deletions_mat', num_deletions_mat, 'num_deletions_pat', num_deletions_pat)

for chrom in chroms:
	positions_mat = chrom_to_positions_mat[chrom]
	positions_mat.add(1)
	positions_mat.add(chrom_lengths[chrom])
	positions_pat = chrom_to_positions_pat[chrom]
	positions_pat.add(1)
	positions_pat.add(chrom_lengths[chrom])

	positions_mat = np.array(sorted(positions_mat))
	positions_pat = np.array(sorted(positions_pat))

	chrom_to_positions_mat[chrom] = positions_mat
	chrom_to_positions_pat[chrom] = positions_pat

	print('chr', chrom, 'intervals_mat', len(chrom_to_positions_mat[chrom]), 'intervals_pat', chrom_to_positions_pat[chrom])
	

# create feature matrix for each chrom
lengths_mat, lengths_pat = [], []
chroms_mat, chroms_pat = [], []
intervals_mat, intervals_pat = [], []

Xs_mat, Xs_pat = [], []
crossovers_mat, crossovers_pat = [], []

Ys_mat, Ys_pat = [], []
deletions_mat, deletions_pat = [], []

for i, chrom in enumerate(chroms):
	# mat
	positions = chrom_to_positions_mat[chrom]
	pos_to_index = dict([(x, i) for i, x in enumerate(positions)])
	intervals_mat.extend([(chrom, start_pos, end_pos) for start_pos, end_pos in zip(positions[:-1], positions[1:])])
	lengths_mat.append(chrom_to_positions_mat[chrom][1:]-chrom_to_positions_mat[chrom][:-1])
	chroms_mat.append(i*np.ones((len(pos_to_index)-1,), dtype=int))

	# mat crossovers
	cos = [co for co in crossovers if co['chrom']==chrom and co['is_mat']]
	crossovers_mat.extend(cos)
	data, row_ind, col_ind = [], [], []
	for row_i, co in enumerate(cos):
		start_index, end_index = pos_to_index[co['start_pos']], pos_to_index[co['end_pos']]
		data.extend([1,]*(end_index-start_index))
		row_ind.extend([row_i,]*(end_index-start_index))
		col_ind.extend(range(start_index, end_index))
	Xs_mat.append(sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(cos), len(pos_to_index)-1)))
	
	# mat deletions
	ds = [d for d in deletions if d['chrom']==chrom and d['is_mat']]
	data, row_ind, col_ind = [], [], []
	for row_i, d in enumerate(ds):
		start_index, end_index = pos_to_index[d['start_pos']], pos_to_index[d['end_pos']]
		data.extend([1,]*(end_index-start_index))
		row_ind.extend([row_i,]*(end_index-start_index))
		col_ind.extend(range(start_index, end_index))
	deletions_mat.extend(ds)
	Ys_mat.append(sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(ds), len(pos_to_index)-1)))
	
	# pat
	positions = chrom_to_positions_pat[chrom]
	pos_to_index = dict([(x, i) for i, x in enumerate(positions)])
	intervals_pat.extend([(chrom, start_pos, end_pos) for start_pos, end_pos in zip(positions[:-1], positions[1:])])
	lengths_pat.append(chrom_to_positions_pat[chrom][1:]-chrom_to_positions_pat[chrom][:-1])
	chroms_pat.append(i*np.ones((len(pos_to_index)-1,), dtype=int))

	# pat crossovers
	cos = [co for co in crossovers if co['chrom']==chrom and co['is_pat']]
	data, row_ind, col_ind = [], [], []
	for row_i, co in enumerate(cos):
		start_index, end_index = pos_to_index[co['start_pos']], pos_to_index[co['end_pos']]
		data.extend([1,]*(end_index-start_index))
		row_ind.extend([row_i,]*(end_index-start_index))
		col_ind.extend(range(start_index, end_index))
	crossovers_pat.extend(cos)
	Xs_pat.append(sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(cos), len(pos_to_index)-1)))
	
	# pat deletions
	ds = [d for d in ds if d['chrom']==chrom and d['is_pat']]
	data, row_ind, col_ind = [], [], []
	for row_i, d in enumerate(ds):
		start_index, end_index = pos_to_index[d['start_pos']], pos_to_index[d['end_pos']]
		data.extend([1,]*(end_index-start_index))
		row_ind.extend([row_i,]*(end_index-start_index))
		col_ind.extend(range(start_index, end_index))
	deletions_pat.extend(ds)
	Ys_pat.append(sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(cos), len(pos_to_index)-1)))
	



with open('recomb_%s/recombination_rates/intervals_mat.txt' % args.dataset, 'w+') as f:
	for chrom, start_pos, end_pos in intervals_mat:
		f.write('%s\t%d\t%d\n' % (chrom, start_pos, end_pos))

with open('recomb_%s/recombination_rates/intervals_pat.txt' % args.dataset, 'w+') as f:
	for chrom, start_pos, end_pos in intervals_pat:
		f.write('%s\t%d\t%d\n' % (chrom, start_pos, end_pos))


X_mat = sparse.block_diag(Xs_mat, format='csr')
X_pat = sparse.block_diag(Xs_pat, format='csr')
Y_mat = sparse.block_diag(Ys_mat, format='csr')
Y_pat = sparse.block_diag(Ys_pat, format='csr')

assert X_mat.shape[0] == len(X_indices_mat)
assert X_pat.shape[0] == len(X_indices_pat)
assert Y_mat.shape[0] == len(X_indices_mat)
assert Y_pat.shape[0] == len(X_indices_pat)

assert np.all(X_mat.sum(axis=1)>0)
assert np.all(X_pat.sum(axis=1)>0)

assert np.all(Y_mat.sum(axis=1)>0)
assert np.all(Y_pat.sum(axis=1)>0)


length_mat = np.hstack(lengths_mat)
length_pat = np.hstack(lengths_pat)
chroms_mat = np.hstack(chroms_mat)
chroms_pat = np.hstack(chroms_pat)

np.save('recomb_%s/recombination_rates/lengths_mat' % args.dataset, length_mat)
np.save('recomb_%s/recombination_rates/lengths_pat' % args.dataset, length_pat)

# estimate recombination rates
def estimate_recombination_rates(X, Y, chrs, length):
	print('Estimating...', X.shape, T.shape, len(chrs), len(length))

	p = cp.Variable(X.shape[1])
	q = cp.Variable()

	# maximize likelihood of multinomial loss...
	expr = cp.sum(cp.log(X@cp.multiply(p, (length/10**6))))/X.shape[0] 
	
	# encourage probabilities to be the same between neighboring intervals (piecewise linear)
	#for i, chrom in enumerate(chroms):
	#	expr += args.lamb * cp.tv(p[chrs==i])
	
	# regularize with maximum entropy
	# -xlog(x)
	# since p is on the scale of Mbp
	# let l be length
	# entropy for each interval is 
	#   -(p/10**6)log(p/10**6)*l
	# = -(lp/10**6)[log(p)-log(10**6)]
	# = -(l/10**6)plogp + (lp/10**6)log(10**6)
	# = entr(p)(l/10**6) + p(l/10**6)log(10**6)
	expr += args.lamb * cp.entr(p)@(length/10**6)
	expr += args.lamb * p@(length/10**6) * np.log(10**6)
	
	# constrain probabilities to be between 0 and 1
	constraints = [p >= 10**-14, p@(length/10**6)==1]


	# now solve
	prob = cp.Problem(cp.Maximize(expr), constraints)
	result = prob.solve(solver='MOSEK', mosek_params={'MSK_IPAR_INTPNT_MAX_ITERATIONS': 500}, verbose=True)
	#result = prob.solve(solver='ECOS', max_iters=200, verbose=True)
	print(prob.status)

	if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
		raise Error('Parameters not fully estimated.')

	return np.clip(p.value, 10**-14, 1) # clip just in case we have some numerical problems

for batch_num in range(args.kfold_crossvalidation):
	test_quads = set(quads[(batch_size*batch_num):(batch_size*(batch_num+1))])
	train_quads = set(quads)-test_quads

	is_mat_train = np.array([tuple(co['child']) in train_quads for co in crossovers_mat])
	is_mat_test = np.array([tuple(co['child']) in test_quads for co in crossovers_mat])
	is_pat_train = np.array([tuple(co['child']) in train_quads for co in crossovers_pat])
	is_pat_test = np.array([tuple(co['child']) in test_quads for co in crossovers_pat])
	assert np.all(is_mat_train | is_mat_test)
	assert np.all(~(is_mat_train & is_mat_test))
	assert np.all(is_pat_train | is_pat_test)
	assert np.all(~(is_pat_train & is_pat_test))

	# MAT
	ps_mat = estimate_recombination_rates(X_mat[is_mat_train, :], chroms_mat, length_mat)
	crossover_ps_mat = X_mat.dot(np.multiply(ps_mat, (length_mat/10**6)))
	crossover_lengths_mat = X_mat.dot(length_mat)
	
	np.save('recomb_%s/recombination_rates/spiky.ps_mat.%g.%d' % (args.dataset, args.lamb, batch_num), ps_mat)
	np.save('recomb_%s/recombination_rates/spiky.crossover_ps_mat_train.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_ps_mat[is_mat_train])
	np.save('recomb_%s/recombination_rates/spiky.crossover_ps_mat_test.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_ps_mat[is_mat_test])
	np.save('recomb_%s/recombination_rates/spiky.crossover_lengths_mat_train.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_lengths_mat[is_mat_train])
	np.save('recomb_%s/recombination_rates/spiky.crossover_lengths_mat_test.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_lengths_mat[is_mat_test])

	# PAT
	ps_pat = estimate_recombination_rates(X_pat[is_pat_train, :], chroms_pat, length_pat)
	crossover_ps_pat = X_pat.dot(np.multiply(ps_pat, (length_pat/10**6)))
	crossover_lengths_pat = X_pat.dot(length_pat)
	
	np.save('recomb_%s/recombination_rates/spiky.ps_pat.%g.%d' % (args.dataset, args.lamb, batch_num), ps_pat)	
	np.save('recomb_%s/recombination_rates/spiky.crossover_ps_pat_train.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_ps_pat[is_pat_train])
	np.save('recomb_%s/recombination_rates/spiky.crossover_ps_pat_test.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_ps_pat[is_pat_test])
	np.save('recomb_%s/recombination_rates/spiky.crossover_lengths_pat_train.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_lengths_pat[is_pat_train])
	np.save('recomb_%s/recombination_rates/spiky.crossover_lengths_pat_test.%g.%d' % (args.dataset, args.lamb, batch_num), crossover_lengths_pat[is_pat_test])


   