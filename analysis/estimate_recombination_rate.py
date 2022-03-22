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
parser.add_argument('lamb', type=float, help='Regularization weight for entropy of combined recombination map.')
parser.add_argument('mat_lamb', type=float, help='Regularization weight for cross entropy between maternal recombination map and combined recombination map.')
parser.add_argument('pat_lamb', type=float, help='Regularization weight for cross entropy between paternal recombination map and combined recombination map.')
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


with open('recomb_%s/sibpairs.json' % args.dataset, 'r') as f:
	sibpairs = json.load(f)

# prune sibpairs so that each child appears in the dataset only once (this is relevant for families with more than 2 children)
random.shuffle(sibpairs)

quads = set()
child_included = set()
for sibpair in sibpairs:
	if sibpair['sibling1'] not in child_included and sibpair['sibling2'] not in child_included:
		quads.add((sibpair['sibling2'], sibpair['sibling1']))
		child_included.add(sibpair['sibling1'])
		child_included.add(sibpair['sibling2'])

print('children with phase information', 2*len(quads))

# pull crossover information
with open('recomb_%s/crossovers.json' % args.dataset, 'r') as f:
	crossovers = [x for x in json.load(f)  if tuple(x['child']) in quads]

# split quads into train/test
quads = list(quads)
random.shuffle(quads)
if args.kfold_crossvalidation == 1:
	batch_size = None
else:
	batch_size = int(np.floor(len(quads)/args.kfold_crossvalidation))
print('batch_size', batch_size)

# pull intervals
chrom_to_positions_mat, chrom_to_positions_pat = defaultdict(set), defaultdict(set)
num_crossovers_mat, num_crossovers_pat = 0, 0

for co in crossovers:
	if co['is_mat']:
		chrom_to_positions_mat[co['chrom']].add(co['start_pos'])
		chrom_to_positions_mat[co['chrom']].add(co['end_pos'])
		chrom_to_positions_pat[co['chrom']].add(co['start_pos'])
		chrom_to_positions_pat[co['chrom']].add(co['end_pos'])

		num_crossovers_mat += 1
		
	if co['is_pat']:
		chrom_to_positions_mat[co['chrom']].add(co['start_pos'])
		chrom_to_positions_mat[co['chrom']].add(co['end_pos'])
		chrom_to_positions_pat[co['chrom']].add(co['start_pos'])
		chrom_to_positions_pat[co['chrom']].add(co['end_pos'])
		num_crossovers_pat += 1

print('num_crossovers_mat', num_crossovers_mat, 'num_crossovers_pat', num_crossovers_pat)

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
Xs_mat, Xs_pat = [], []
lengths_mat, lengths_pat = [], []
chroms_mat, chroms_pat = [], []
intervals_mat, intervals_pat = [], []
crossovers_mat, crossovers_pat = [], []

chrom_index_offset_mat, chrom_index_offset_pat = 0, 0

for i, chrom in enumerate(chroms):
	positions = chrom_to_positions_mat[chrom]
	pos_to_index = dict([(x, i) for i, x in enumerate(positions)])
	cos = [co for co in crossovers if co['chrom']==chrom and co['is_mat']]
	data, row_ind, col_ind = [], [], []
	for row_i, co in enumerate(cos):
		start_index, end_index = pos_to_index[co['start_pos']], pos_to_index[co['end_pos']]
		data.extend([1,]*(end_index-start_index))
		row_ind.extend([row_i,]*(end_index-start_index))
		col_ind.extend(range(start_index, end_index))
	crossovers_mat.extend(cos)
	intervals_mat.extend([(chrom, start_pos, end_pos) for start_pos, end_pos in zip(positions[:-1], positions[1:])])
	Xs_mat.append(sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(cos), len(pos_to_index)-1)))
	lengths_mat.append(chrom_to_positions_mat[chrom][1:]-chrom_to_positions_mat[chrom][:-1])
	chroms_mat.append(i*np.ones((len(pos_to_index)-1,), dtype=int))
	chrom_index_offset_mat += len(pos_to_index)-1

	positions = chrom_to_positions_pat[chrom]
	pos_to_index = dict([(x, i) for i, x in enumerate(positions)])
	cos = [co for co in crossovers if co['chrom']==chrom and co['is_pat']]
	data, row_ind, col_ind = [], [], []
	for row_i, co in enumerate(cos):
		start_index, end_index = pos_to_index[co['start_pos']], pos_to_index[co['end_pos']]
		data.extend([1,]*(end_index-start_index))
		row_ind.extend([row_i,]*(end_index-start_index))
		col_ind.extend(range(start_index, end_index))
	crossovers_pat.extend(cos)
	intervals_pat.extend([(chrom, start_pos, end_pos) for start_pos, end_pos in zip(positions[:-1], positions[1:])])
	Xs_pat.append(sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(cos), len(pos_to_index)-1)))
	lengths_pat.append(chrom_to_positions_pat[chrom][1:]-chrom_to_positions_pat[chrom][:-1])
	chroms_pat.append(i*np.ones((len(pos_to_index)-1,), dtype=int))
	chrom_index_offset_pat += len(pos_to_index)-1

with open('recomb_%s/recombination_rates/intervals_mat.%g.%g.%g.txt' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), 'w+') as f:
	for chrom, start_pos, end_pos in intervals_mat:
		f.write('%s\t%d\t%d\n' % (chrom, start_pos, end_pos))

with open('recomb_%s/recombination_rates/intervals_pat.%g.%g.%g.txt' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), 'w+') as f:
	for chrom, start_pos, end_pos in intervals_pat:
		f.write('%s\t%d\t%d\n' % (chrom, start_pos, end_pos))


X_mat = sparse.block_diag(Xs_mat, format='csr')
X_pat = sparse.block_diag(Xs_pat, format='csr')


assert np.all(X_mat.sum(axis=1)>0)
assert np.all(X_pat.sum(axis=1)>0)


length_mat = np.hstack(lengths_mat)
length_pat = np.hstack(lengths_pat)
chroms_mat = np.hstack(chroms_mat)
chroms_pat = np.hstack(chroms_pat)

np.save('recomb_%s/recombination_rates/lengths_mat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), length_mat)
np.save('recomb_%s/recombination_rates/lengths_pat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), length_pat)

# estimate recombination rates
def estimate_recombination_rates(X_mat, X_pat, chrs, length):
	print('Estimating...', X_mat.shape, X_pat.shape, len(chrs), len(length))

	assert X_mat.shape[1] == X_pat.shape[1]

	p = cp.Variable(X_mat.shape[1])
	p_mat = cp.Variable(X_mat.shape[1])
	p_pat = cp.Variable(X_pat.shape[1])

	# maximize likelihood of multinomial loss...
	expr = cp.sum(cp.log(X_mat@p_mat))/X_mat.shape[0] 
	expr += cp.sum(cp.log(X_pat@p_pat))/X_pat.shape[0] 

	# regularize with maximum entropy
	# -xlog(x)
	# let l be length
	# entropy for each interval is 
	#   -plog(p)*l
	# = -lplog(p) - lplog(l) + lplog(l)
	# = -lplog(lp) + lplog(l)

	expr += args.lamb * cp.sum(cp.entr(p))
	expr += args.lamb * p@cp.log(length)

	expr -= args.mat_lamb * cp.sum(cp.rel_entr(p_mat, p)) 
	expr -= args.pat_lamb * cp.sum(cp.rel_entr(p_pat, p))
	
	
	# constrain probabilities to be between 0 and 1
	constraints = [p>=0, p_mat>=0, p_pat>=0, cp.sum(p)==1, cp.sum(p_mat)==1, cp.sum(p_pat)==1]


	# now solve
	prob = cp.Problem(cp.Maximize(expr), constraints)
	result = prob.solve(solver='MOSEK', mosek_params={'MSK_IPAR_INTPNT_MAX_ITERATIONS': 500}, verbose=True)
	#result = prob.solve(solver='ECOS', max_iters=200, verbose=True)
	print(prob.status)

	if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
		raise Error('Parameters not fully estimated.')

	return np.clip(p_mat.value, 0, 1), np.clip(p_pat.value, 0, 1) # clip just in case we have some numerical problems

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
	ps_mat, ps_pat = estimate_recombination_rates(X_mat[is_mat_train, :], X_pat[is_pat_train, :], chroms_mat, length_mat)
	crossover_ps_mat = X_mat.dot(ps_mat)
	crossover_lengths_mat = X_mat.dot(length_mat)
	
	np.save('recomb_%s/recombination_rates/ps_mat.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb, batch_num), ps_mat/length_mat)
	np.save('recomb_%s/recombination_rates/crossover_ps_mat_train.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb, batch_num), crossover_ps_mat[is_mat_train])
	np.save('recomb_%s/recombination_rates/crossover_ps_mat_test.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb, batch_num), crossover_ps_mat[is_mat_test])
	np.save('recomb_%s/recombination_rates/crossover_lengths_mat_train.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb, batch_num), crossover_lengths_mat[is_mat_train])
	np.save('recomb_%s/recombination_rates/crossover_lengths_mat_test.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb, batch_num), crossover_lengths_mat[is_mat_test])

	# PAT
	crossover_ps_pat = X_pat.dot(ps_pat)
	crossover_lengths_pat = X_pat.dot(length_pat)
	
	np.save('recomb_%s/recombination_rates/ps_pat.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lambbatch_num), ps_pat/length_pat)	
	np.save('recomb_%s/recombination_rates/crossover_ps_pat_train.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lambbatch_num), crossover_ps_pat[is_pat_train])
	np.save('recomb_%s/recombination_rates/crossover_ps_pat_test.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lambbatch_num), crossover_ps_pat[is_pat_test])
	np.save('recomb_%s/recombination_rates/crossover_lengths_pat_train.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lambbatch_num), crossover_lengths_pat[is_pat_train])
	np.save('recomb_%s/recombination_rates/crossover_lengths_pat_test.%g.%g.%g.%d' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lambbatch_num), crossover_lengths_pat[is_pat_test])

# MAT
ps_mat, ps_pat = estimate_recombination_rates(X_mat, X_pat, chroms_mat, length_mat)
crossover_ps_mat = X_mat.dot(ps_mat)
crossover_lengths_mat = X_mat.dot(length_mat)

np.save('recomb_%s/recombination_rates/ps_mat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), ps_mat/length_mat)
np.save('recomb_%s/recombination_rates/crossover_ps_mat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), crossover_ps_mat)
np.save('recomb_%s/recombination_rates/crossover_lengths_mat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), crossover_lengths_mat)

# PAT
crossover_ps_pat = X_pat.dot(ps_pat)
crossover_lengths_pat = X_pat.dot(length_pat)

np.save('recomb_%s/recombination_rates/ps_pat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), ps_pat/length_pat)	
np.save('recomb_%s/recombination_rates/crossover_ps_pat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), crossover_ps_pat)
np.save('recomb_%s/recombination_rates/crossover_lengths_pat.%g.%g.%g' % (args.dataset, args.lamb, args.mat_lamb, args.pat_lamb), crossover_lengths_pat)


   