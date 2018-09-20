import sys
import time
from os import listdir
import gzip

from collections import Counter, Hashable, defaultdict
from itertools import chain, product

import numpy as np
from scipy import sparse
import random


# Run locally with python3 phase/phase_chromosomeX.py 3 data/160826.ped split_gen_miss phased

chrom = 'X'
m = int(sys.argv[1])
ped_file = sys.argv[2]
data_dir = sys.argv[3]
out_dir = sys.argv[4]
batch_size = None if len(sys.argv) < 8 else int(sys.argv[5])
batch_num = None if len(sys.argv) < 8 else int(sys.argv[6])

sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
coord_file = '%s/chr.%s.gen.coordinates.npy' % (data_dir,  chrom)
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s' % chrom) in f and 'gen.npz' in f])

fam_output_file = '%s/chr.%s.familysize.%s.families.txt' % (out_dir, chrom, m)
phase_output_file = '%s/chr.%s.familysize.%s.phased.txt' % (out_dir, chrom, m)

if batch_size is not None:
	batch_offset = batch_size*batch_num
	fam_output_file = fam_output_file[:-4] + str(batch_num) + '.txt'
	phase_output_file = phase_output_file[:-4] + str(batch_num) + '.txt'


PAR1X_end = 2699520
PAR2X_start = 154931044

# From GRCh37.p13 https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh37.p13
chrom_lengths = {
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
chrom_length = chrom_lengths[chrom]

# genotype (pred, obs): cost
g_cost = {
	(-1, -1): 0,
	(-1, 0): 1,
	(-1, 1): 1,
	(-1, 2): 1,
	(0, -1): 0,
	(0, 0): 0,
	(0, 1): 1,
	(0, 2): 2,
	(1, -1): 0,
	(1, 0): 1,
	(1, 1): 0,
	(1, 2): 1,
	(2, -1): 0,
	(2, 0): 2,
	(2, 1): 1,
	(2, 2): 0
}

# pull families with sequence data
with open(sample_file, 'r') as f:
	sample_ids = [line.strip() for line in f]
sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])

# pull families from ped file
sample_id_to_sex = dict()
families = dict()
with open(ped_file, 'r') as f:	
    for line in f:
        pieces = line.strip().split('\t')
        fam_id, child_id, f_id, m_id = pieces[:4]

        if child_id in sample_ids and f_id in sample_ids and m_id in sample_ids:
        	if (fam_id, m_id, f_id) not in families:
        		families[(fam_id, m_id, f_id)] = [m_id, f_id]
        	families[(fam_id, m_id, f_id)].append(child_id)
        	sample_id_to_sex[child_id] = pieces[4]

# randomly permute children
families = dict([(k, x[:2]+random.sample(x[2:], len(x)-2)) for k, x in families.items()])
family_to_indices = dict([(fid, [sample_id_to_index[x] for x in vs]) for fid, vs in families.items()])
print('families with sequence data', len(families))

families_of_this_size = [(fkey, ind_indices) for fkey, ind_indices in family_to_indices.items() if len(ind_indices) == m]
print('families of size %d: %d' % (m, len(families_of_this_size)))

# limit to batch
if batch_size is not None:
	family_keys = set(sorted([x[0] for x in families_of_this_size])[batch_offset:(batch_size+batch_offset)])
	families_of_this_size = [(k, v) for k, v in families_of_this_size if k in family_keys]
	print('families of size %d: %d' % (m, len(families_of_this_size)))

# inheritance states
#
# for parents:
# (0, 1) -> deletion on parental1
# (1, 0) -> deletion on parental2
# (1, 1) -> normal
# 
# for children:
# (0, 0) -> m1p1
# (0, 1) -> m1p2
# (1, 0) -> m2p1
# (1, 1) -> m2p2

if m >= 5:
	inheritance_states = np.array(list(product(*([[0, 1]]*(2*m)))), dtype=np.int8)
else:
	inheritance_states = np.array([x for x in product(*([[0, 1]]*(2*m))) if x[4]==0], dtype=np.int8)
state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states)])
p, state_len = inheritance_states.shape
print('inheritance states', inheritance_states.shape)

# perfect match genotypes
pm_gen_to_index = dict()
pm_gen_indices = []
for s in inheritance_states:
    if s[4] == 0 and s[5] == 0:
        anc_pos = [[-1] if s[i] == 0 else [0, 1] for i in range(4)]
        anc_variants = np.array(list(product(*anc_pos)), dtype=np.int8)
        pred_gens = np.zeros((anc_variants.shape[0], m), dtype=np.int8)

        # mom
        # deletion
        pred_gens[(anc_variants[:, 0]==-1) & (anc_variants[:, 1]==-1), 0] = -1
        pred_gens[(anc_variants[:, 0]==-1) & (anc_variants[:, 1]==0), 0] = 0
        pred_gens[(anc_variants[:, 0]==-1) & (anc_variants[:, 1]==1), 0] = 2
        pred_gens[(anc_variants[:, 0]==0) & (anc_variants[:, 1]==-1), 0] = 0
        pred_gens[(anc_variants[:, 0]==1) & (anc_variants[:, 1]==-1), 0] = 2
        # normal
        pred_gens[(anc_variants[:, 0]==0) & (anc_variants[:, 1]==0), 0] = 0
        pred_gens[(anc_variants[:, 0]==1) & (anc_variants[:, 1]==1), 0] = 2
        pred_gens[(anc_variants[:, 0]==0) & (anc_variants[:, 1]==1), 0] = 1
        pred_gens[(anc_variants[:, 0]==1) & (anc_variants[:, 1]==0), 0] = 1

        # dad
        # deletion
        pred_gens[(anc_variants[:, 2]==-1) & (anc_variants[:, 3]==-1), 1] = -1
        pred_gens[(anc_variants[:, 2]==-1) & (anc_variants[:, 3]==0), 1] = 0
        pred_gens[(anc_variants[:, 2]==-1) & (anc_variants[:, 3]==1), 1] = 2
        pred_gens[(anc_variants[:, 2]==0) & (anc_variants[:, 3]==-1), 1] = 0
        pred_gens[(anc_variants[:, 2]==1) & (anc_variants[:, 3]==-1), 1] = 2
        # normal
        pred_gens[(anc_variants[:, 2]==0) & (anc_variants[:, 3]==0), 1] = 0
        pred_gens[(anc_variants[:, 2]==1) & (anc_variants[:, 3]==1), 1] = 2
        pred_gens[(anc_variants[:, 2]==0) & (anc_variants[:, 3]==1), 1] = 1
        pred_gens[(anc_variants[:, 2]==1) & (anc_variants[:, 3]==0), 1] = 1

        # children
        for index in range(m-2):
            mat, pat = s[(4+(2*index)):(6+(2*index))]

            # deletion
            pred_gens[(anc_variants[:, mat]==-1) & (anc_variants[:, 2+pat]==-1), 2+index] = -1
            pred_gens[(anc_variants[:, mat]==-1) & (anc_variants[:, 2+pat]==0), 2+index] = 0
            pred_gens[(anc_variants[:, mat]==-1) & (anc_variants[:, 2+pat]==1), 2+index] = 2
            pred_gens[(anc_variants[:, mat]==0) & (anc_variants[:, 2+pat]==-1), 2+index] = 0
            pred_gens[(anc_variants[:, mat]==1) & (anc_variants[:, 2+pat]==-1), 2+index] = 2
            # normal
            pred_gens[(anc_variants[:, mat]==0) & (anc_variants[:, 2+pat]==0), 2+index] = 0
            pred_gens[(anc_variants[:, mat]==1) & (anc_variants[:, 2+pat]==1), 2+index] = 2
            pred_gens[(anc_variants[:, mat]==0) & (anc_variants[:, 2+pat]==1), 2+index] = 1
            pred_gens[(anc_variants[:, mat]==1) & (anc_variants[:, 2+pat]==0), 2+index] = 1

        unique_pred_gens = set(map(tuple, pred_gens))
        for pg in unique_pred_gens:
            if pg not in pm_gen_to_index:
                pm_gen_to_index[pg] = len(pm_gen_to_index)
        pm_gen_indices.append([pm_gen_to_index[pg] for pg in unique_pred_gens])

pm_gen = np.zeros((len(pm_gen_to_index), m), dtype=np.int8)
for pm, i in pm_gen_to_index.items():
	pm_gen[i, :] = pm
print('perfect matches', pm_gen.shape, len(pm_gen_indices), Counter([len(v) for v in pm_gen_indices]))

# losses are symmetrical to parental chromosome swaps
full_loss_indices = np.zeros((p,), dtype=int)
loss_state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states[(inheritance_states[:, 4]==0) & (inheritance_states[:, 5]==0), :])])
for i, s in enumerate(inheritance_states):
	new_s = -np.ones((state_len,), dtype=np.int8)
	if s[4] == 0:
		new_s[:2] = s[:2]
		new_s[np.arange(4, state_len, 2)] = s[np.arange(4, state_len, 2)]
	else:
		new_s[:2] = s[[1, 0]]
		new_s[np.arange(4, state_len, 2)] = 1-s[np.arange(4, state_len, 2)]
	if s[5] == 0:
		new_s[2:4] = s[2:4]
		new_s[np.arange(5, state_len, 2)] = s[np.arange(5, state_len, 2)]
	else:
		new_s[2:4] = s[[3, 2]]
		new_s[np.arange(5, state_len, 2)] = 1-s[np.arange(5, state_len, 2)]
	full_loss_indices[i] = loss_state_to_index[tuple(new_s)]

genotypes = np.array(list(product(*[[-1, 0, 1, 2]]*m)), dtype=np.int8)
genotype_to_index = dict([(tuple(x), i) for i, x in enumerate(genotypes)])
q = genotypes.shape[0]
print('genotypes', genotypes.shape)

losses = np.zeros((int(p/4) if m>=5 else p, q), dtype=np.int8)
already_calculated = np.zeros((q,), dtype=bool)
def calculate_loss(gen): 
	gen_index = genotype_to_index[tuple(gen)]
	if not already_calculated[gen_index]:
		s = np.zeros((len(pm_gen_to_index),), dtype=np.int8)
		for pm, i in pm_gen_to_index.items():
			s[i] = sum([g_cost[(pred, obs)] for pred, obs in zip(pm, gen)])
	    
		for i, indices in enumerate(pm_gen_indices):
			losses[i, gen_index] = np.min(s[indices])
		already_calculated[gen_index] = True
	return losses[full_loss_indices, gen_index]

# pull genotype data from .npz
indices_of_interest = sum([v for k, v in families_of_this_size], [])
old_index_to_new_index = dict([(ind, i) for (i, ind) in enumerate(indices_of_interest)])
families_of_this_size = [(k, [old_index_to_new_index[x] for x in v]) for (k, v) in families_of_this_size]
	
# use only "cleaned" variants - must be SNPs
coordinates = np.load(coord_file)
snp_positions = coordinates[:, 1]
snp_indices = coordinates[:, 2]==1

snp_positions = snp_positions[snp_indices]
min_position, max_position = snp_positions[0], snp_positions[-1]
print('chrom shape only SNPs', snp_positions.shape[0])


with open(fam_output_file, 'w+') as famf, open(phase_output_file, 'w+') as statef:
	# write headers
	famf.write('family_id\tmother_id\tfather_id\t' + '\t'.join(['child%d_id' % i for i in range(1, m-1)]) + '\n')
	statef.write('\t'.join(['family_id', 'state_id', 'm1_state', 'm2_state', 'p1_state', 'p2_state',
		'\t'.join(['child%d_%s_state' % ((i+1), c) for i, c in product(range(m-2), ['m', 'p'])]),
		'start_pos', 'end_pos', 'start_family_index', 'end_family_index' 'pos_length', 'family_index_length']) + '\n')

	# phase each family
	for fkey, ind_indices in families_of_this_size:
		inds = families[fkey]
		ind_indices = [sample_id_to_index[x] for x in inds]
		print('family', fkey)

		# pull genotype data for this family

		#load from npz
		data = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file))[ind_indices, :] for gen_file in gen_files]).A
		data = data[:, snp_indices]

		n = 2*snp_positions.shape[0]+1
		family_genotypes = np.zeros((m, n), dtype=np.int8)
		family_genotypes[:, np.arange(1, n-1, 2)] = data

		# if we see two missing entries in a row, mark the middle interval as possibly missing/possibly homref (-2)
		family_genotypes[family_genotypes<0] = -1
		for i in range(m):
			double_missing = np.where((data[i, 1:]==-1) & (data[i, :-1]==-1))[0]
			family_genotypes[i, (2*double_missing)+2] = -1

		family_snp_positions = np.zeros((n, 2), dtype=np.int)
		family_snp_positions[0, 0] = 0
		family_snp_positions[np.arange(0, n-2, 2), 1] = snp_positions-1
		family_snp_positions[np.arange(1, n-1, 2), 0] = snp_positions-1
		family_snp_positions[np.arange(1, n-1, 2), 1] = snp_positions
		family_snp_positions[np.arange(2, n, 2), 0] = snp_positions
		family_snp_positions[-1, 1] = chrom_lengths[chrom]

		# remove unnecessary ref positions
		haslength = np.where(family_snp_positions[:, 0]!=family_snp_positions[:, 1])[0]
		family_genotypes = family_genotypes[:, haslength]
		family_snp_positions = family_snp_positions[haslength, :]

		# aggregate identical genotypes
		rep_indices = np.where(np.any(family_genotypes[:, 1:]!=family_genotypes[:, :-1], axis=0))[0]
		n = rep_indices.shape[0]+1

		new_family_genotypes = np.zeros((m, n), dtype=np.int8)
		new_family_genotypes[:, :-1] = family_genotypes[:, rep_indices]
		new_family_genotypes[:, -1] = family_genotypes[:, -1]

		new_family_snp_positions = np.zeros((n, 2), dtype=np.int)
		new_family_snp_positions[0, 0] = family_snp_positions[0, 0]
		new_family_snp_positions[:-1, 1] = family_snp_positions[rep_indices, 1]
		new_family_snp_positions[1:, 0] = family_snp_positions[rep_indices+1, 0]
		new_family_snp_positions[-1, 1] = family_snp_positions[-1, 1]

		family_genotypes, family_snp_positions = new_family_genotypes, new_family_snp_positions

		mult_factor = family_snp_positions[:, 1] - family_snp_positions[:, 0]

		# transition matrix
		# only allow one shift at a time
		shift_costs = [10]*4 + [500]*(2*(m-2))

		transitions = [[] for i in range(p)]
		transition_costs = [[] for i in range(p)]
		for i, state in enumerate(inheritance_states):
			for delstate in list(product(*[[0, 1]]*4)):
				new_state = tuple(delstate) + tuple(state[4:])
				if new_state in state_to_index:
					new_index = state_to_index[new_state]
					transitions[i].append(new_index)
					transition_costs[i].append(sum([shift_costs[j] for j, (old_s, new_s) in enumerate(zip(state[:4], delstate)) if old_s != new_s]))

			# allow a single recombination event
			for j in range(4, inheritance_states.shape[1]):
				new_state = tuple(1-x if k == j else x for k, x in enumerate(state))
				if new_state in state_to_index:
					new_index = state_to_index[new_state]
					transitions[i].append(new_index)
					transition_costs[i].append(shift_costs[j])
		            
		# We need different transition matrices depending on where we are on the X-chrom

		# first, decide which states are allowed outside the PAR
		out_par_states = []
		for i, s in enumerate(inheritance_states):
			# females inherit p1, males inherit p2
			inheritance_ok = True
			for j, child in enumerate(inds[2:]):
				if sample_id_to_sex[child] == '2':
					# female child
					inheritance_ok = inheritance_ok and (s[(2*j)+5] == 0)
				if sample_id_to_sex[child] == '1':
					# male child
					inheritance_ok = inheritance_ok and (s[(2*j)+5] == 1)
		            
			# deletion on p2
			if inheritance_ok and s[3]==0:
				out_par_states.append(i)
		r = len(out_par_states)
		out_par_state_to_index = dict([(i, x) for x, i in enumerate(out_par_states)])
		#print('states allowed outside PAR', r, '\nstates allowed inside PAR', p)

		                
		# in PAR transitions
		in_par_transitions = np.array(transitions)
		in_par_transition_costs = np.array(transition_costs)

		starting_state = [1, 1, 1, 1, 0]
		if sample_id_to_sex[inds[2]] == '2':
			# first child is female
			starting_state.append(0)
		else:
			starting_state.append(1)
		zero_transition_costs = np.zeros((p,), dtype=int)
		for i, ss in enumerate(starting_state):
			zero_transition_costs[inheritance_states[:, i] != ss] += shift_costs[i]
		#print('in PAR transitions', in_par_transitions.shape, np.max(in_par_transitions))

		# out PAR transitions
		out_par_transitions, out_par_transition_costs = [], []
		for i in range(p):
			if i in out_par_states:
				trans, costs = [], []
				for t, c in zip(in_par_transitions[i, :], in_par_transition_costs[i, :]):
					if t in out_par_states:
						trans.append(out_par_state_to_index[t])
						costs.append(c)
		                
				out_par_transitions.append(trans)
				out_par_transition_costs.append(costs)
		out_par_transitions = np.array(out_par_transitions)
		out_par_transition_costs = np.array(out_par_transition_costs)
		#print('out PAR transitions', out_par_transitions.shape, np.max(out_par_transitions))
	
		# from PAR transitions
		from_par_transitions = np.zeros((r, p), dtype=int)
		to_par_transitions = np.zeros((p, r), dtype=int)
		from_to_par_transition_costs = np.zeros((r, p), dtype=int)

		for i in range(p):
			from_par_transitions[:, i] = i
		    
		for i in range(r):
			to_par_transitions[:, i] = i
		    
		for i in range(p):
			for j in range(r):
				from_to_par_transition_costs[j, i] = np.abs(inheritance_states[i, :]-inheritance_states[out_par_states[j], :]).dot(shift_costs)

		#print('from PAR transitions', from_par_transitions.shape, np.max(from_par_transitions))
		#print('to PAR transitions', to_par_transitions.shape, np.max(to_par_transitions))

		# viterbi
		last_in_par1 = np.argmax(family_snp_positions[:, 0] > PAR1X_end)
		first_in_par2 = np.argmax(family_snp_positions[:, 0] > PAR2X_start)
		#print('Last index in PAR1', last_in_par1)
		#print('First index in PAR2', first_in_par2)

		PAR1_v_cost = np.zeros((p, last_in_par1), dtype=int)
		outPAR_v_cost = np.zeros((r, first_in_par2-last_in_par1), dtype=int)
		PAR2_v_cost = np.zeros((p, n-first_in_par2), dtype=int)

		# forward sweep
		prev_time = time.time()

		# first step, break symmetry
		# we enforce that the chromosome starts with child1 (0, ) and no deletions or duplications
		pos_gen = tuple(family_genotypes[:, 0])
		loss = calculate_loss(pos_gen).astype(int)
		PAR1_v_cost[:, 0] = mult_factor[0]*loss + zero_transition_costs

		# PAR1
		for j in range(1, last_in_par1): 
			pos_gen = tuple(family_genotypes[:, j])
			loss = calculate_loss(pos_gen).astype(int)
			PAR1_v_cost[:, j] = np.min(PAR1_v_cost[in_par_transitions, j-1] + in_par_transition_costs, axis=1) + mult_factor[j]*loss

		# Transition out of PAR
		pos_gen = tuple(family_genotypes[:, last_in_par1])
		loss = calculate_loss(pos_gen).astype(int)[out_par_states]
		outPAR_v_cost[:, 0] = np.min(PAR1_v_cost[from_par_transitions, -1] + from_to_par_transition_costs, axis=1) + mult_factor[last_in_par1]*loss

		# Out of PAR
		for j in range(last_in_par1+1, first_in_par2): 
			pos_gen = tuple(family_genotypes[:, j])
			loss = calculate_loss(pos_gen).astype(int)[out_par_states]
			outPAR_v_cost[:, j-last_in_par1] = np.min(outPAR_v_cost[out_par_transitions, j-1-last_in_par1] + out_par_transition_costs, axis=1) + mult_factor[j]*loss

		# Transition into PAR
		pos_gen = tuple(family_genotypes[:, first_in_par2])
		loss = calculate_loss(pos_gen).astype(int)
		PAR2_v_cost[:, 0] = np.min(outPAR_v_cost[to_par_transitions, -1] + from_to_par_transition_costs.T, axis=1) + mult_factor[first_in_par2]*loss

		# PAR2
		for j in range(first_in_par2+1, n): 
			pos_gen = tuple(family_genotypes[:, j])
			loss = calculate_loss(pos_gen).astype(int)
			PAR2_v_cost[:, j-first_in_par2] = np.min(PAR2_v_cost[in_par_transitions, j-1-first_in_par2] + in_par_transition_costs, axis=1) + mult_factor[j]*loss

		print('Forward sweep complete', time.time()-prev_time, 'sec') 

		# write header to file
		famf.write('%s\t%s\n' % ('.'.join(fkey), '\t'.join(inds)))
		famf.flush()

		# backward sweep
		prev_time = time.time()
		final_states = -np.ones((state_len, n), dtype=int)

		# choose best paths
		# we enforce that the chromosome ends with no deletions
		num_forks = 0
		no_delstates = np.all(inheritance_states[:, :4]==1, axis=1)
		min_value = np.min(PAR2_v_cost[no_delstates, -1])
		paths = np.where((PAR2_v_cost[:, -1]==min_value) & no_delstates)[0]
		print('Num solutions', paths.shape, inheritance_states[paths, :])

		# combine path states into a single state (unknown values represented with -1)
		if paths.shape[0] == 1:
			final_states[:, -1] = inheritance_states[paths[0], :]
		else:
			num_forks += 1
			path_states = inheritance_states[paths, :]
			known_indices = np.all(path_states == path_states[0, :], axis=0)
			final_states[known_indices, -1] = path_states[0, known_indices]

		# now work backwards 

		# PAR2
		for j in reversed(range(PAR2_v_cost.shape[1]-1)):

			# traceback
			total_cost = PAR2_v_cost[in_par_transitions[paths, :], j] + in_par_transition_costs[paths, :]
			min_value = np.min(total_cost, axis=1)
			new_paths = set()
			for i, k in enumerate(paths):
				min_indices = in_par_transitions[k, np.where(total_cost[i, :] == min_value[i])[0]]
				new_paths.update(min_indices.tolist())
			paths = np.asarray(list(new_paths), dtype=int)

			# combine path states into a single state (unknown values represented with -1)
			if paths.shape[0] == 1:
				final_states[:, j+first_in_par2] = inheritance_states[paths[0], :]
			else:
				num_forks += 1
				path_states = inheritance_states[paths, :]
				known_indices = np.all(path_states == path_states[0, :], axis=0)
				final_states[known_indices, j+first_in_par2] = path_states[0, known_indices]
		
		# Transition out of PAR
		total_cost = outPAR_v_cost[to_par_transitions[paths, :], 0] + from_to_par_transition_costs.T[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = to_par_transitions[k, np.where(total_cost[i, :] == min_value[i])[0]]
			new_paths.update([out_par_states[i] for i in min_indices.tolist()])
		paths = np.asarray(list(new_paths), dtype=int)

		# combine path states into a single state (unknown values represented with -1)
		if paths.shape[0] == 1:
			final_states[:, first_in_par2-1] = inheritance_states[paths[0], :]
		else:
			num_forks += 1
			path_states = inheritance_states[paths, :]
			known_indices = np.all(path_states == path_states[0, :], axis=0)
			final_states[known_indices, first_in_par2-1] = path_states[0, known_indices]

		# Out of PAR
		for j in reversed(range(outPAR_v_cost.shape[1]-1)):

			# traceback
			paths = [out_par_state_to_index[p] for p in paths]
			total_cost = outPAR_v_cost[out_par_transitions[paths, :], j] + out_par_transition_costs[paths, :]
			min_value = np.min(total_cost, axis=1)
			new_paths = set()
			for i, k in enumerate(paths):
				min_indices = out_par_transitions[k, np.where(total_cost[i, :] == min_value[i])[0]]
				new_paths.update([out_par_states[i] for i in min_indices.tolist()])
			paths = np.asarray(list(new_paths), dtype=int)

			# combine path states into a single state (unknown values represented with -1)
			if paths.shape[0] == 1:
				final_states[:, j+last_in_par1] = inheritance_states[paths[0], :]
			else:
				num_forks += 1
				path_states = inheritance_states[paths, :]
				known_indices = np.all(path_states == path_states[0, :], axis=0)
				final_states[known_indices, j+last_in_par1] = path_states[0, known_indices]
		
		# Transition into PAR
		paths = [out_par_state_to_index[p] for p in paths]
		total_cost = PAR1_v_cost[from_par_transitions[paths, :], 0] + from_to_par_transition_costs[paths, :]
		min_value = np.min(total_cost, axis=1)
		new_paths = set()
		for i, k in enumerate(paths):
			min_indices = from_par_transitions[k, np.where(total_cost[i, :] == min_value[i])[0]]
			new_paths.update(min_indices.tolist())
		paths = np.asarray(list(new_paths), dtype=int)

		# combine path states into a single state (unknown values represented with -1)
		if paths.shape[0] == 1:
			final_states[:, last_in_par1-1] = inheritance_states[paths[0], :]
		else:
			num_forks += 1
			path_states = inheritance_states[paths, :]
			known_indices = np.all(path_states == path_states[0, :], axis=0)
			final_states[known_indices, last_in_par1-1] = path_states[0, known_indices]
		    
		# PAR1
		for j in reversed(range(PAR1_v_cost.shape[1]-1)):

			# traceback
			total_cost = PAR1_v_cost[in_par_transitions[paths, :], j] + in_par_transition_costs[paths, :]
			min_value = np.min(total_cost, axis=1)
			new_paths = set()
			for i, k in enumerate(paths):
				min_indices = in_par_transitions[k, np.where(total_cost[i, :] == min_value[i])[0]]
				new_paths.update(min_indices.tolist())
			paths = np.asarray(list(new_paths), dtype=int)

			# combine path states into a single state (unknown values represented with -1)
			if paths.shape[0] == 1:
				final_states[:, j] = inheritance_states[paths[0], :]
			else:
				num_forks += 1
				path_states = inheritance_states[paths, :]
				known_indices = np.all(path_states == path_states[0, :], axis=0)
				final_states[known_indices, j] = path_states[0, known_indices]

		print('Num positions in fork', num_forks)
		print('Backward sweep complete', time.time()-prev_time, 'sec') 

		# if a parental chromosome isn't inherited, then we don't know if it has a deletion
		maternal_indices = range(4, state_len, 2)
		paternal_indices = range(5, state_len, 2)

		final_states[0, np.all(final_states[maternal_indices, :]!=0, axis=0)] = -1
		final_states[1, np.all(final_states[maternal_indices, :]!=1, axis=0)] = -1
		final_states[2, np.all(final_states[paternal_indices, :]!=0, axis=0)] = -1
		final_states[3, np.all(final_states[paternal_indices, :]!=1, axis=0)] = -1

		# write to file
		change_indices = [-1] + np.where(np.any(final_states[:, 1:]!=final_states[:, :-1], axis=0))[0].tolist()
		for j in range(1, len(change_indices)):
			s_start, s_end = change_indices[j-1]+1, change_indices[j]
			#assert np.all(final_states[:, s_start] == final_states[:, s_end])
			statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
						'.'.join(fkey), 
						'\t'.join(map(str, final_states[:, s_start])), 
						family_snp_positions[s_start, 0]+1, family_snp_positions[s_end, 1],
						s_start, s_end, 
						family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0], 
						s_end-s_start+1))

		# last entry
		s_start, s_end = change_indices[-1]+1, family_snp_positions.shape[0]-1
		#assert np.all(final_states[:, s_start] == final_states[:, s_end])
		statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
					'.'.join(fkey), 
					'\t'.join(map(str, final_states[:, s_start])), 
					family_snp_positions[s_start, 0]+1, family_snp_positions[s_end, 1],
					s_start, s_end, 
					family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0]+1, 
					s_end-s_start+1))
		statef.flush()	

		print('Write to file complete')
