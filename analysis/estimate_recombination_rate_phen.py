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
parser.add_argument('kfold_crossvalidation', type=int, help='Number of fold to be used for cross validation. 1 means no cross validation')
parser.add_argument('lamb', type=float, help='Regularization weight.')
parser.add_argument('phase_dirs', type=str, nargs='+', help='Directories with phase information.')
parser.add_argument('--ped_files', type=str, nargs='+', help='Files with ped information.')

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

# pull ped data
def pull_phenotype_ped(ped_file):
	sample_to_sex = dict()
	sample_to_affected = dict()
	parents_to_children = defaultdict(list)
	with open(ped_file, 'r') as f:
		for line in f:
			pieces = line.strip().split('\t')
			sample_to_sex[pieces[1]] = pieces[4]
			sample_to_affected[pieces[1]] = pieces[5]
			if pieces[2] != '0' and pieces[3] != '0':
				parents_to_children[(pieces[3], pieces[2])].append(pieces[1])
	return sample_to_affected, sample_to_sex, parents_to_children

sample_to_affected = dict()
for ped_file in args.ped_files:
	s_to_a, _, _ = pull_phenotype_ped(ped_file)
	sample_to_affected.update(s_to_a)

# pull crossover information, preferentially pull sibpairs with same phenotype
quad_to_crossovers = defaultdict(list)
already_included = set()
for phase_dir in args.phase_dirs:
	print('loading phase for', phase_dir)
	for file in sorted(os.listdir(phase_dir)):
		if file.endswith('.crossovers.json'):
			with open('%s/%s' % (phase_dir, file), 'r') as f:
				cos = json.load(f)
				child1, child2 = cos[0]['child']
				if len(cos)>=250:
					#print('Are they related?', cos[0]['child'][0], cos[0]['child'][1], file)
					pass
				elif child1 in already_included or child2 in already_included:
					#print('Child(ren) already included in analysis', cos[0]['child'][0], cos[0]['child'][1], file)
					pass
				elif child1 in sample_to_affected and child2 in sample_to_affected and sample_to_affected[child1]==sample_to_affected[child2]:
					for co in cos:
						co['num_affected'] = int(sample_to_affected[child1]=='2') + int(sample_to_affected[child2]=='2')
						quad_to_crossovers[tuple(co['child'])].append(co)
					already_included.add(co['child'][0])
					already_included.add(co['child'][1])

for phase_dir in args.phase_dirs:
	print('loading phase for', phase_dir)
	for file in sorted(os.listdir(phase_dir)):
		if file.endswith('.crossovers.json'):
			with open('%s/%s' % (phase_dir, file), 'r') as f:
				cos = json.load(f)
				child1, child2 = cos[0]['child']
				if len(cos)>=250:
					#print('Are they related?', cos[0]['child'][0], cos[0]['child'][1], file)
					pass
				elif child1 in already_included or child2 in already_included:
					#print('Child(ren) already included in analysis', cos[0]['child'][0], cos[0]['child'][1], file)
					pass
				elif child1 in sample_to_affected and child2 in sample_to_affected:
					for co in cos:
						co['num_affected'] = int(sample_to_affected[child1]=='2') + int(sample_to_affected[child2]=='2')
						quad_to_crossovers[tuple(co['child'])].append(co)
					already_included.add(co['child'][0])
					already_included.add(co['child'][1])
            
print('children with phase information', 2*len(quad_to_crossovers))
print('num_affected 0=%d, 1=%d, 2=%d' % (len([x for x in quad_to_crossovers.keys() if int(sample_to_affected[x[0]]=='2')+int(sample_to_affected[x[1]]=='2')==0]),
										 len([x for x in quad_to_crossovers.keys() if int(sample_to_affected[x[0]]=='2')+int(sample_to_affected[x[1]]=='2')==1]),
										 len([x for x in quad_to_crossovers.keys() if int(sample_to_affected[x[0]]=='2')+int(sample_to_affected[x[1]]=='2')==2])))

# split quads into train/test
quads = sorted(quad_to_crossovers.keys())
random.shuffle(quads)
if args.kfold_crossvalidation == 1:
	batch_size = None
else:
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


all_positions = sorted(all_positions_mat | all_positions_pat | set([(chrom, 1) for chrom in chroms]) | set([(chrom, chrom_lengths[str(chrom)]) for chrom in chroms]))
pos_to_index = dict([(x, i) for i, x in enumerate(all_positions)])
lengths = np.array([0 if all_positions[i][0] != all_positions[i+1][0] else all_positions[i+1][1]-all_positions[i][1] for i in range(len(all_positions)-1)])
print('num_crossovers', num_crossovers_mat+num_crossovers_pat, 'positions', len(all_positions))

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
	return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(crossovers), len(pos_to_index)-1))

# estimate recombination rates
def estimate_recombination_rates(X_mat, X_pat, phen_mat, phen_pat, lengths, positions):
	print('Estimating...', X_mat.shape, X_pat.shape)

	p_mat_aut = cp.Variable(X_mat.shape[1])
	p_pat_aut = cp.Variable(X_pat.shape[1])
	p_mat_typ = cp.Variable(X_mat.shape[1])
	p_pat_typ = cp.Variable(X_pat.shape[1])

	# maximize likelihood of multinomial loss...
	expr = (cp.sum(cp.log(X_mat[phen_mat==2, :]@p_mat_aut))+\
		cp.sum(cp.log(0.5*X_mat[phen_mat==1, :]@p_mat_aut + 0.5*X_mat[phen_mat==1, :]@p_mat_typ))+\
		cp.sum(cp.log(X_mat[phen_mat==0, :]@p_mat_typ)))/X_mat.shape[0] + \
		   (cp.sum(cp.log(X_pat[phen_pat==2, :]@p_pat_aut))+\
		cp.sum(cp.log(0.5*X_pat[phen_pat==1, :]@p_pat_aut + 0.5*X_pat[phen_pat==1, :]@p_pat_typ))+\
		cp.sum(cp.log(X_pat[phen_pat==0, :]@p_pat_typ)))/X_pat.shape[0]

	# and regularize the difference between neighboring intervals (within each chromosome)
	current_chrom = positions[0][0]
	current_chrom_start = 0
	for i in range(1, len(positions)):
		if positions[i][0] != current_chrom:
			# link nearby positions
			expr -= cp.tv(args.lamb*p_mat_aut[current_chrom_start:i]/lengths[current_chrom_start:i])
			expr -= cp.tv(args.lamb*p_pat_aut[current_chrom_start:i]/lengths[current_chrom_start:i])
			expr -= cp.tv(args.lamb*p_mat_typ[current_chrom_start:i]/lengths[current_chrom_start:i])
			expr -= cp.tv(args.lamb*p_pat_typ[current_chrom_start:i]/lengths[current_chrom_start:i])

			# link mat/pat
			expr -= cp.tv(args.lamb*(p_mat_aut[current_chrom_start:i]-p_pat_aut[current_chrom_start:i])/lengths[current_chrom_start:i])
			expr -= cp.tv(args.lamb*(p_mat_typ[current_chrom_start:i]-p_pat_typ[current_chrom_start:i])/lengths[current_chrom_start:i])

			# link aut/typ
			expr -= cp.tv(args.lamb*(p_mat_aut[current_chrom_start:i]-p_mat_typ[current_chrom_start:i])/lengths[current_chrom_start:i])
			expr -= cp.tv(args.lamb*(p_pat_aut[current_chrom_start:i]-p_pat_typ[current_chrom_start:i])/lengths[current_chrom_start:i])

			current_chrom, current_chrom_start = positions[i][0], i
	expr -= cp.tv(args.lamb*p_mat_aut[current_chrom_start:i]/lengths[current_chrom_start:i])
	expr -= cp.tv(args.lamb*p_pat_aut[current_chrom_start:i]/lengths[current_chrom_start:i])
	expr -= cp.tv(args.lamb*p_mat_typ[current_chrom_start:i]/lengths[current_chrom_start:i])
	expr -= cp.tv(args.lamb*p_pat_typ[current_chrom_start:i]/lengths[current_chrom_start:i])

	# link mat/pat
	expr -= cp.tv(args.lamb*(p_mat_aut[current_chrom_start:i]-p_pat_aut[current_chrom_start:i])/lengths[current_chrom_start:i])
	expr -= cp.tv(args.lamb*(p_mat_typ[current_chrom_start:i]-p_pat_typ[current_chrom_start:i])/lengths[current_chrom_start:i])

	# link aut/typ
	expr -= cp.tv(args.lamb*(p_mat_aut[current_chrom_start:i]-p_mat_typ[current_chrom_start:i])/lengths[current_chrom_start:i])
	expr -= cp.tv(args.lamb*(p_pat_aut[current_chrom_start:i]-p_pat_typ[current_chrom_start:i])/lengths[current_chrom_start:i])

	# now solve
	print('length range', np.max(lengths), np.min(lengths))
	prob = cp.Problem(cp.Maximize(expr), [p_mat_aut >= lengths*(10**-11), p_mat_aut<=1, cp.sum(p_mat_aut)==1,
										  p_pat_aut >= lengths*(10**-11), p_pat_aut<=1, cp.sum(p_pat_aut)==1,
										  p_mat_typ >= lengths*(10**-11), p_mat_typ<=1, cp.sum(p_mat_typ)==1,
										  p_pat_typ >= lengths*(10**-11), p_pat_typ<=1, cp.sum(p_pat_typ)==1,])
	result = prob.solve(solver='MOSEK', mosek_params={'MSK_IPAR_INTPNT_MAX_ITERATIONS': 500}, verbose=True)
	#result = prob.solve(solver='ECOS', max_iters=200, verbose=True)
	print(prob.status)

	if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
		raise Error('Parameters not fully estimated.')

	p_mat_aut = np.clip([v for v in p_mat_aut.value], 0, 1) # clip just in case we have some numerical problems
	p_pat_aut = np.clip([v for v in p_pat_aut.value], 0, 1) # clip just in case we have some numerical problems
	p_mat_typ = np.clip([v for v in p_mat_typ.value], 0, 1) # clip just in case we have some numerical problems
	p_pat_typ = np.clip([v for v in p_pat_typ.value], 0, 1) # clip just in case we have some numerical problems

	return p_mat_aut, p_pat_aut, p_mat_typ, p_pat_typ

for batch_num in range(args.kfold_crossvalidation):
	if args.kfold_crossvalidation==1:
		test_quads = set(quads)
		train_quads = set(quads)
	else:
		test_quads = set(quads[(batch_size*batch_num):(batch_size*(batch_num+1))])
		train_quads = set(quads)-test_quads

	if args.kfold_crossvalidation == 1:
		with open('%s.train_quads.%g.txt' % (args.out_file, args.lamb), 'w+') as f:
			for quad in train_quads:
				f.write('\t'.join(quad) + '\n')
	else:
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

	X_mat_train = X_from_crossovers(crossovers_mat_train, pos_to_index)
	X_mat_test = X_from_crossovers(crossovers_mat_test, pos_to_index)
	X_pat_train = X_from_crossovers(crossovers_pat_train, pos_to_index)
	X_pat_test = X_from_crossovers(crossovers_pat_test, pos_to_index)

	phen_mat_train = np.array([co['num_affected'] for co in crossovers_mat_train])
	phen_pat_train = np.array([co['num_affected'] for co in crossovers_pat_train])
	phen_mat_test = np.array([co['num_affected'] for co in crossovers_mat_test])
	phen_pat_test = np.array([co['num_affected'] for co in crossovers_pat_test])

	#sparse.save_npz('%s.X_mat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_train)
	#sparse.save_npz('%s.X_mat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_test)
	#sparse.save_npz('%s.X_pat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_train)
	#sparse.save_npz('%s.X_pat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_test)

	ps_mat_aut, ps_pat_aut, ps_mat_typ, ps_pat_typ = estimate_recombination_rates(X_mat_train, X_pat_train, phen_mat_train, phen_pat_train, lengths, all_positions)
	
	def predict_with_phen(X, ps_aut, ps_typ, phen):
		pred = np.zeros((X.shape[0],))
		pred[phen==0] = X[phen==0, :].dot(ps_typ)
		pred[phen==1] = 0.5*X[phen==1, :].dot(ps_aut) + 0.5*X[phen==1, :].dot(ps_typ)
		pred[phen==2] = X[phen==2, :].dot(ps_aut)
		return pred

	if args.kfold_crossvalidation == 1:
		np.save('%s.ps_mat_aut.%g' % (args.out_file, args.lamb), ps_mat_aut)
		np.save('%s.ps_mat_typ.%g' % (args.out_file, args.lamb), ps_mat_typ)
		np.save('%s.crossover_ps_mat_train.%g' % (args.out_file, args.lamb), predict_with_phen(X_mat_train, ps_mat_aut, ps_mat_typ, phen_mat_train))
		np.save('%s.crossover_lengths_mat_aut_train.%g' % (args.out_file, args.lamb), X_mat_train.dot(lengths))

		np.save('%s.ps_pat_aut.%g' % (args.out_file, args.lamb), ps_pat_aut)
		np.save('%s.ps_pat_typ.%g' % (args.out_file, args.lamb), ps_pat_typ)
		np.save('%s.crossover_ps_pat_train.%g' % (args.out_file, args.lamb), predict_with_phen(X_pat_train, ps_pat_aut, ps_pat_typ, phen_pat_train))
		np.save('%s.crossover_lengths_pat_train.%g' % (args.out_file, args.lamb), X_pat_train.dot(lengths))
	else:
		np.save('%s.ps_mat_aut.%g.%d' % (args.out_file, args.lamb, batch_num), ps_mat_aut)
		np.save('%s.ps_mat_typ.%g.%d' % (args.out_file, args.lamb, batch_num), ps_mat_typ)
		np.save('%s.crossover_ps_mat_train.%g.%d' % (args.out_file, args.lamb, batch_num), predict_with_phen(X_mat_train, ps_mat_aut, ps_mat_typ, phen_mat_train))
		np.save('%s.crossover_ps_mat_test.%g.%d' % (args.out_file, args.lamb, batch_num), predict_with_phen(X_mat_test, ps_mat_aut, ps_mat_typ, phen_mat_test))
		np.save('%s.crossover_lengths_mat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_train.dot(lengths))
		np.save('%s.crossover_lengths_mat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_mat_test.dot(lengths))

		np.save('%s.ps_pat_aut.%g.%d' % (args.out_file, args.lamb, batch_num), ps_pat_aut)
		np.save('%s.ps_pat_typ.%g.%d' % (args.out_file, args.lamb, batch_num), ps_pat_typ)
		np.save('%s.crossover_ps_pat_train.%g.%d' % (args.out_file, args.lamb, batch_num), predict_with_phen(X_pat_train, ps_pat_aut, ps_pat_typ, phen_pat_train))
		np.save('%s.crossover_ps_pat_test.%g.%d' % (args.out_file, args.lamb, batch_num), predict_with_phen(X_pat_test, ps_pat_aut, ps_pat_typ, phen_pat_test))
		np.save('%s.crossover_lengths_pat_train.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_train.dot(lengths))
		np.save('%s.crossover_lengths_pat_test.%g.%d' % (args.out_file, args.lamb, batch_num), X_pat_test.dot(lengths))

           