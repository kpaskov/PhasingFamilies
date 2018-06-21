import sys
import time
from os import listdir
import gzip

from collections import Counter
from itertools import chain, product

import numpy as np
from scipy import sparse
import random


# Run locally with python3 de_novo_deletions.py 22 4 160826.ped split_gen phased

chrom = sys.argv[1]
family_size = int(sys.argv[2])
ped_file = sys.argv[3]
data_dir = sys.argv[4]
out_dir = sys.argv[5]

sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
variant_file = '%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom)
clean_file = '%s/clean_indices_%s.txt' % (data_dir, chrom) 
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s' % chrom) in f and 'gen.npz' in f])
phase_family_file = '%s/chr.%s.familysize.%d.families.txt' % (phase_dir, chrom, j)
phase_file = '%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, j)

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
pos_to_index = dict([(x, i) for i, x in enumerate(snp_positions)])

whole_chrom = whole_chrom[:, snp_indices]
m, n = whole_chrom.shape
print('chrom shape only SNPs', m, n)

# inheritance states
#
# for each child:
# (0, 0) -> normal
# (0, 1) -> de novo paternal deletion
# (1, 0) -> de novo maternal deletion
# (1, 1) -> de novo maternal+paternal deletion

inheritance_states = np.array(list(product(*[[0, 1]]*(2*(family_size-2)))), dtype=np.int8)
state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states)])
p = inheritance_states.shape[0]
print('inheritance states', inheritance_states.shape)

# transition matrix
shift_cost = 10
s = np.tile(np.sum(inheritance_states, axis=1), (p, 1))
transition_costs = shift_cost*(s + s.T - 2*inheritance_states.dot(inheritance_states.T))
print('transitions', transition_costs)

with open('%s/chr.%s.familysize.%s.families.txt' % (out_dir, chrom, family_size), 'w+') as famf, open('%s/chr.%s.familysize.%s.phased.txt' % (out_dir, chrom, family_size), 'w+') as statef:
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
		family_indices = np.where(pos_to_genindex != genotype_to_index[(0,)*m])[0]
		family_genotypes = family_genotypes[:, family_indices]
		family_snp_positions = snp_positions[family_indices]
		n = pos_to_genindex.shape[0]
		print('family chrom shape', m, n)

		# pull inheritance patterns
inherit_patterns = -2*np.ones((m, 2, n), dtype=np.int8)
with open(phase_family_file, 'r')  as famf, open(phase_file, 'r')  as phasef:
	next(famf) # skip header
	next(phasef) # skip header
	        
	fam_pieces = (None,)
	for line in phasef:
		pieces = line.strip().split('\t')
		family_key = pieces[0]
		inheritance_state = [None if x == '*' else int(x) for x in pieces[1:(1+(j*2))]]
		del_state = [0 if x is None else x for x in inheritance_state[:4]]
		start_pos, end_pos = [int(x) for x in pieces[(1+(j*2)):(3+(j*2))]]
		start_index, end_index = pos_to_index[start_pos], pos_to_index[end_pos]
	            
		# make sure we're on the right family
		while family_key != fam_pieces[0]:
			fam_pieces = next(famf).strip().split('\t')
			fam_individuals = fam_pieces[1:(1+j)]
			fam_indices = [None if (ind not in sample_id_to_index) or (sample_id_to_index[ind] not in old_index_to_new_index) else old_index_to_new_index[sample_id_to_index[ind]] for ind in fam_individuals]
	               
		if fam_indices[0] is not None:
			inherit_patterns[fam_indices[0], start_index:(end_index+1)] = sum(del_state[:2])
		if fam_indices[1] is not None:
			deletions[fam_indices[1], start_index:(end_index+1)] = sum(del_state[2:4])
	            
		for k, child_index in enumerate(fam_indices[2:]):
			if child_index is not None:
				mat, pat = inheritance_state[(4+(2*k)):(6+(2*k))]
				if mat is not None and pat is not None:
					deletions[child_index, start_index:(end_index+1)] = (del_state[mat]+del_state[2+pat])
				elif (mat is None) and (pat is not None) and (del_state[0] == del_state[1]):
					deletions[child_index, start_index:(end_index+1)] = (del_state[0]+del_state[2+pat])
				elif (pat is None) and (mat is not None) and (del_state[2] == del_state[3]):
					deletions[child_index, start_index:(end_index+1)] = (del_state[mat]+del_state[2])
				elif (del_state[0] == del_state[1]) and (del_state[2] == del_state[3]):
					deletions[child_index, start_index:(end_index+1)] = (del_state[0]+del_state[2])

# genotype-inheritance


m = family_size
print('Family size', m)

genotypes = np.array(list(product(*[[-1, 0, 1, 2]]*m)), dtype=np.int8)
genotype_to_index = dict([(tuple(x), i) for i, x in enumerate(genotypes)])
q = genotypes.shape[0]
print('genotypes', genotypes.shape)



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

		# viterbi
		v_cost = np.zeros((p, n+1), dtype=int)
		#v_traceback = np.zeros((p, n+1), dtype=int)
		
		# forward sweep
		prev_time = time.time()

		for j in range(n): 
		    v_cost[:, j+1] = np.min(np.tile(v_cost[:, j], (p, 1)) + transition_costs, axis=1) + losses[:, pos_to_genindex[j]]

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
			total_cost = v_cost[transitions[paths, :], index-1] + transition_costs[paths, :]
			min_value = np.min(total_cost, axis=1)
			new_states = set()
			for i, k in enumerate(paths):
				# get best tracebacks
				min_indices = transitions[k, np.where(total_cost[i, :] <= min_value[i])[0]]	
				new_states.update(min_indices.tolist())
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
