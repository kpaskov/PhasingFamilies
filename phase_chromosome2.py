import sys
import time
from os import listdir
import gzip

from collections import Counter
from itertools import chain, product

import numpy as np
from scipy import sparse
import random


# Run locally with python3 phase_chromosome2.py 22 all 160826.ped split_gen

chrom = sys.argv[1]
family_size = int(sys.argv[2])
ped_file = sys.argv[3] #'data/v34.forCompoundHet.ped'
data_dir = sys.argv[4]

sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, 'X' if chrom.startswith('PAR') else chrom)
variant_file = '%s/chr.%s.gen.variants.txt.gz' % (data_dir, 'X' if chrom.startswith('PAR') else chrom)
clean_file = '%s/clean_indices_%s.txt' % (data_dir, 'X' if chrom.startswith('PAR') else chrom) 
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s' % ('X' if chrom.startswith('PAR') else chrom)) in f and 'gen.npz' in f])

# genotype (pred, obs): cost
g_cost = {
	(-1, -1): 0,
	(-1, 0): 1,
	(-1, 1): 1,
	(-1, 2): 1,
	(0, -1): 1,
	(0, 0): 0,
	(0, 1): 1,
	(0, 2): 2,
	(1, -1): 1,
	(1, 0): 1,
	(1, 1): 0,
	(1, 2): 1,
	(2, -1): 1,
	(2, 0): 2,
	(2, 1): 1,
	(2, 2): 0
}

PAR1X = (60001, 2699520)
PAR2X = (154931044, 155260560)

# pull families with sequence data
with open(sample_file, 'r') as f:
	sample_ids = [line.strip() for line in f]
sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])

# pull families from ped file
families = dict()
with open(ped_file, 'r') as f:	
    for line in f:
        pieces = line.strip().split('\t')
        fam_id, child_id, f_id, m_id = pieces[0:4]

        if child_id in sample_ids and f_id in sample_ids and m_id in sample_ids:
        	if (fam_id, m_id, f_id) not in families:
        		families[(fam_id, m_id, f_id)] = [m_id, f_id]
        	families[(fam_id, m_id, f_id)].append(child_id)

# randomly permute parents and children (separately)
family_to_mom_dad = dict([(k, tuple(x[:2])) for k, x in families.items()])
families = dict([(k, random.sample(x[:2], 2)+random.sample(x[2:], len(x)-2)) for k, x in families.items()])
family_to_indices = dict([(fid, [sample_id_to_index[x] for x in vs]) for fid, vs in families.items()])
family_to_index = dict([(fid, i) for i, fid in enumerate(families.keys())])

print('families with sequence data', len(families))

families_of_this_size = [(fkey, ind_indices) for fkey, ind_indices in family_to_indices.items() if len(ind_indices) == family_size]
print('families of size %d: %d' % (family_size, len(families_of_this_size)))

# ancestral_variants (m1, m2, p1, p2)
anc_variants = np.array(list(product(*[[0, 1]]*4)), dtype=np.int8)
anc_variant_to_index = dict([(tuple(x), i) for i, x in enumerate(anc_variants)])
print('ancestral variants', anc_variants.shape)

# pull genotype data from .npz
indices_of_interest = sum([v for k, v in families_of_this_size], [])
old_index_to_new_index = dict([(ind, i) for (i, ind) in enumerate(indices_of_interest)])
families_of_this_size = [(k, [old_index_to_new_index[x] for x in v]) for (k, v) in families_of_this_size]
whole_chrom = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file))[indices_of_interest,:] for gen_file in gen_files])
m, n = whole_chrom.shape
print('chrom shape', m, n, type(whole_chrom))
	
# use only "cleaned" variants - must be SNPs and missingness in parents can't be sex-biased
snp_indices = []
snp_positions = []
with open(clean_file, 'r') as f:
    for i, line in enumerate(f):
        index, position = line.strip().split('\t')
        snp_indices.append(int(index))
        snp_positions.append(int(position))
snp_positions = np.array(snp_positions)

whole_chrom = whole_chrom[:, snp_indices]
m, n = whole_chrom.shape
print('chrom shape only SNPs', m, n)

# If we're looking at one of the PAR, restrict X to appropriate region
if chrom == 'PAR1':
	PAR1X_indices = np.where(np.logical_and(snp_positions >= PAR1X[0], snp_positions <= PAR1X[1]))[0]
	whole_chrom = whole_chrom[:, PAR1X_indices]
	snp_positions = snp_positions[PAR1X_indices]
elif chrom == 'PAR2':
	PAR2X_indices = np.where(np.logical_and(snp_positions >= PAR2X[0], snp_positions <= PAR2X[1]))[0]
	whole_chrom = whole_chrom[:, PAR2X_indices]
	snp_positions = snp_positions[PAR2X_indices]

m = family_size
print('Family size', m)
# inheritance states
#
# for parents:
# (0, 0) -> normal
# (0, 1) -> deletion on parental2
# (1, 0) -> deletion on parental1
# (1, 1) -> deletion on parental1 and parental2
# 
# for children:
# (0, 0) -> m1p1
# (0, 1) -> m1p2
# (1, 0) -> m2p1
# (1, 1) -> m2p2

if family_size >= 5:
	inheritance_states = np.array(list(product(*[[0, 1]]*(2*m))), dtype=np.int8)
else:
	inheritance_states = np.array([x for x in product(*[[0, 1]]*(2*m)) if x[4] == 0 and x[5] == 0], dtype=np.int8)
state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states)])
p = inheritance_states.shape[0]
print('inheritance states', inheritance_states.shape)

# genotypes
genotypes = np.array(list(product(*[[-1, 0, 1, 2]]*m)), dtype=np.int8)
genotype_to_index = dict([(tuple(x), i) for i, x in enumerate(genotypes)])
q = genotypes.shape[0]
print('genotypes', genotypes.shape)

# transition matrix
# only allow one shift at a time
shift_costs = [10]*4 + [500]*(2*(m-2))

transitions = [[] for i in range(p)]
transition_costs = [[] for i in range(p)]
for i, state in enumerate(inheritance_states):
	# allow multiple deletion transitions
	for delstate in list(product(*[[0, 1]]*4)):
		new_state = tuple(delstate) + tuple(state[4:])
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
            
transitions = np.array(transitions)
transition_costs = np.array(transition_costs)
print('transitions', transitions.shape)

# loss matrix
losses = np.zeros((p, q), dtype=np.int16)
for i, s in enumerate(inheritance_states):
	state_losses = np.zeros((q, anc_variants.shape[0]), dtype=np.int16)

	# mom
	if s[0] == 0 and s[1] == 0:
		pred_gens = anc_variants[:, 0] + anc_variants[:, 1]
	elif s[0] == 0:
		pred_gens = 2*anc_variants[:, 0]
	elif s[1] == 0:
		pred_gens = 2*anc_variants[:, 1]
	else:
		pred_gens = -1*np.ones((anc_variants.shape[0],))
	for obs_gen in [-1, 0, 1, 2]:
		state_losses[genotypes[:, 0]==obs_gen, :] += [g_cost[(pred_gen, obs_gen)] for pred_gen in pred_gens]
        
	# dad
	if s[2] == 0 and s[3] == 0:
		pred_gens = anc_variants[:, 2] + anc_variants[:, 3]
	elif s[2] == 0:
		pred_gens = 2*anc_variants[:, 2]
	elif s[3] == 0:
		pred_gens = 2*anc_variants[:, 3]
	else:
		pred_gens = -1*np.ones((anc_variants.shape[0],))
	for obs_gen in [-1, 0, 1, 2]:
		state_losses[genotypes[:, 1]==obs_gen, :] += [g_cost[(pred_gen, obs_gen)] for pred_gen in pred_gens]
      
	# children
	for index in range(m-2):
		mat, pat = s[(4+(2*index)):(6+(2*index))]

		if s[mat] == 0 and s[2+pat] == 0:
			pred_gens = anc_variants[:, mat] + anc_variants[:, 2+pat]
		elif s[mat] == 0:
			pred_gens = 2*anc_variants[:, mat]
		elif s[2+pat] == 0:
			pred_gens = 2*anc_variants[:, 2+pat]
		else:
			pred_gens = -1*np.ones((anc_variants.shape[0],))
		for obs_gen in [-1, 0, 1, 2]:
			state_losses[genotypes[:, 2+index]==obs_gen, :] += [g_cost[(pred_gen, obs_gen)] for pred_gen in pred_gens]

	losses[i, :] = np.min(state_losses, axis=1)

print('losses', losses.shape, losses)

with open('phased/chr.%s.familysize.%s.families.txt' % (chrom, family_size), 'w+') as famf, open('phased/chr.%s.familysize.%s.phased.txt' % (chrom, family_size), 'w+') as statef:
	# write headers
	famf.write('\t'.join(['family_id', 'mother_id', 'father_id', 
		'\t'.join(['child%d_id' % (i+1) for i in range(0, family_size-2)]), 
		'mother_vcfindex', 'father_vcfindex',
		'\t'.join(['child%d_vcfindex' % (i+1) for i in range(0, family_size-2)])]) + '\n')
	statef.write('\t'.join(['family_id', 'state_id', 'm1_state', 'm2_state', 'p1_state', 'p2_state',
		'\t'.join(['child%d_%s_state' % ((i+1), c) for i, c in product(range(family_size-2), ['m', 'p'])]),
		'start_pos', 'end_pos', 'start_index', 'end_index', 'start_family_index', 'end_family_index' 'pos_length', 'index_length', 'family_index_length']) + '\n')

	# phase each family
	for fkey, ind_indices in families_of_this_size:
		family_index = family_to_index[fkey]
		print('family', fkey, family_index)

		# pull genotype data for this family
		family_genotypes = whole_chrom[ind_indices, :].A

		# filter out all_hom_ref
		pos_to_genindex = np.asarray([genotype_to_index[tuple(x)] for x in family_genotypes.T])
		family_indices = np.where(pos_to_genindex != genotype_to_index[(0,)*m])[0]
		pos_to_genindex = pos_to_genindex[family_indices]
		family_snp_positions = snp_positions[family_indices]
		n = pos_to_genindex.shape[0]
		print('family chrom shape', m, n)

		# viterbi
		v_cost = np.zeros((p, n+1), dtype=int)
		v_traceback = np.zeros((p, n+1), dtype=int)
		v_num_traceback = np.zeros((p, n+1), dtype=int)

		# store traceback in column sparse format (to allow multiple pointers)
		v_traceback_indices, v_traceback_indptr = [], [0]
		
		# forward sweep
		prev_time = time.time()
		v_cost[:, 0] = [2*s[4]+s[5] for s in inheritance_states]
		v_traceback[:, 0] = -1
		for j in range(n): 
		    total_cost = v_cost[transitions, j] + transition_costs
		    min_index = np.argmin(total_cost, axis=1)
		    min_value = total_cost[range(p), min_index]
		    v_traceback[:, j+1] = transitions[range(p), min_index]
		    v_cost[:, j+1] = min_value + losses[:, pos_to_genindex[j]]

		    # check for multiple tracebacks
		    v_num_traceback[:, j+1] = np.sum(total_cost == np.repeat(np.reshape(min_value, (p, 1)), transitions.shape[1], axis=1), axis=1)
		    #v_num_traceback[:, j+1] = np.sum(np.apply_along_axis(lambda x: x == min_value, 0, total_cost), axis=1)
		print('Forward sweep complete', time.time()-prev_time, 'sec') 

		# write header to file
		if family_to_mom_dad[fkey] == tuple(families[fkey][:2]):
			# mom comes first already
			new_order = list(range(len(families[fkey])))
			new_state_order = list(range(2*len(families[fkey])))
		else:
			new_order = [1, 0] + list(range(2, len(families[fkey])))
			new_state_order = [2, 3, 0, 1] + list(range(4, 2*len(families[fkey])))

		famf.write('%s\t%s\t%s\n' % ('.'.join(fkey), '\t'.join([families[fkey][i] for i in new_order]), '\t'.join(map(str, [ind_indices[i] for i in new_order]))))
		famf.flush()

		# backward sweep
		prev_time = time.time()
		
		# choose best path
		k = np.argmin(v_cost[:, n])
		print('Num solutions', np.sum(v_cost[:, n]==v_cost[k, n]))
		paths = [k]
		prev_state = tuple(inheritance_states[k, :])
		prev_state_end = n-1
		
		num_forks = 0
		index = n
		while index > 0:
			# traceback
			new_states = set()
			for k in paths:
				if v_num_traceback[k, index] > 1:
					# get alternate tracebacks
					total_cost = v_cost[transitions[k, :], index-1] + transition_costs[k, :]
					min_value = np.min(total_cost)
					min_indices = transitions[k, np.where(total_cost == min_value)[0]]	
					new_states.update(min_indices.tolist())
				else:
					new_states.add(v_traceback[k, index])
			new_states = list(new_states)

			# combine into a single state (with missing values)
			new_state = tuple(inheritance_states[new_states[0], :])
			if len(new_states) > 1:
				num_forks += 1
				for k in new_states[1:]:
					new_state = tuple(['*' if x != y else x for x, y in zip(new_state, tuple(inheritance_states[k, :]))])

			# write to file
			if prev_state != new_state:
				s_start, s_end = index, prev_state_end
				statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
					'.'.join(fkey), 
					'\t'.join([str(prev_state[i]) for i in new_state_order]), 
					family_snp_positions[s_start], family_snp_positions[s_end], 
					family_indices[s_start], family_indices[s_end], 
					s_start, s_end, 
					family_snp_positions[s_end]-family_snp_positions[s_start], 
					family_indices[s_end]-family_indices[s_start], 
					s_end-s_start))
				prev_state = new_state
				prev_state_end = index-1

			index -= 1
			paths = list(new_states)

		# last state
		s_start, s_end = 0, prev_state_end
		statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
					'.'.join(fkey), 
					'\t'.join([str(prev_state[i]) for i in new_state_order]), 
					family_snp_positions[s_start], family_snp_positions[s_end], 
					family_indices[s_start], family_indices[s_end], 
					s_start, s_end, 
					family_snp_positions[s_end]-family_snp_positions[s_start], 
					family_indices[s_end]-family_indices[s_start], 
					s_end-s_start))
		statef.flush()	

		print('Num positions in fork', num_forks)
		print('Backward sweep complete', time.time()-prev_time, 'sec') 
