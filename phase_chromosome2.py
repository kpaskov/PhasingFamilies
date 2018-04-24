import sys
import time
from os import listdir
import gzip

from collections import Counter
from itertools import chain, product

import numpy as np
from scipy import sparse

chrom = sys.argv[1]
ped_file = sys.argv[2] #'data/v34.forCompoundHet.ped'
data_dir = sys.argv[3]
sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
variant_file = '%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom)

# constants
g_neighbors = {0: [1],
	           1: [0, 2],
	           2: [1],
	          -1: [0, 1, 2]}

g_equivalents = {0: [0, -1],
	             1: [1, -1],
	             2: [2, -1],
	            -1: [-1]}

# ancestral_variants (m1, m2, p1, p2)
anc_variants = np.array(list(product(*[[0, 1]]*4)), dtype=int)
anc_variant_to_index = dict([(tuple(x), i) for i, x in enumerate(anc_variants)])
print('ancestral variants', anc_variants.shape)

# pull families with sequence data
with open(sample_file, 'r') as f:
	sample_ids = [line.strip() for line in f]
sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])

families = dict()
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        fam_id, child_id, f_id, m_id = pieces[0:4]

        if child_id in sample_ids and f_id in sample_ids and m_id in sample_ids:
        	if (fam_id, m_id, f_id) not in families:
        		families[(fam_id, m_id, f_id)] = [m_id, f_id]
        	families[(fam_id, m_id, f_id)].append(child_id)

families = dict(list(families.items())[:10])
family_to_indices = dict([(fid, [sample_id_to_index[x] for x in vs]) for fid, vs in families.items()])
family_to_index = dict([(fid, i) for i, fid in enumerate(families.keys())])

print('families with sequence data', len(families))
print('family sizes', Counter([len(x) for x in families.values()]))

# pull genotype data from .npz
gen_files = sorted([f for f in listdir('split_gen') if ('chr.%s' % chrom) in f and 'gen.npz' in f])
whole_chrom = sparse.hstack([sparse.load_npz('split_gen/%s' % gen_file) for gen_file in gen_files])
m, n = whole_chrom.shape
print('chrom shape', m, n)

# discard variants that aren't SNPs
snp_indices = []
snp_positions = []
with gzip.open(variant_file, 'rt') as f:
    for i, line in enumerate(f):
        pieces = line.strip().split('\t')
        if len(pieces[3]) == 1 and len(pieces[4]) == 1 and pieces[3] != '.' and pieces[4] != '.':
            snp_indices.append(i)
            snp_positions.append(int(pieces[1]))
whole_chrom = whole_chrom[:, snp_indices]
snp_positions = np.array(snp_positions)
m, n = whole_chrom.shape
print('chrom shape only SNPs', m, n)

all_states = np.zeros((len(families), n), dtype=int)
all_losses = np.zeros((len(families), n), dtype=int)

# do one family size at a time so we can reuse preprocessed matrices
for m in Counter([len(x) for x in families.values()]).keys():
#for m in [4, 5, 6, 3, 7, 8, 9, 10]:
#for m in [4]:
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

	inheritance_states = np.array(list(product(*[[0, 1]]*(2*m))), dtype=int)
	state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states)])
	p = inheritance_states.shape[0]
	print('inheritance states', inheritance_states.shape)

	# genotypes
	genotypes = np.array(list(product(*[[-1, 0, 1, 2]]*m)), dtype=np.int8)
	genotype_to_index = dict([(tuple(x), i) for i, x in enumerate(genotypes)])
	q = genotypes.shape[0]
	print('genotypes', genotypes.shape)

	# map genotypes to neighbors and equivalents
	genotype_to_neighbors = []
	genotype_to_equivalents = []
	for g in genotypes:
	    neighbor_gs = []
	    for j in range(m):
	        for new_entry in g_neighbors[g[j]]:
	            new_genotype = tuple(new_entry if k == j else x for k, x in enumerate(g))
	            neighbor_gs.append(genotype_to_index[new_genotype])
	    genotype_to_neighbors.append(neighbor_gs)   
	    genotype_to_equivalents.append([genotype_to_index[tuple(x)] for x in product(*[g_equivalents[x] for x in g])]) 
	
	print('genotype to equivalents', Counter([len(x) for x in genotype_to_equivalents]))    
	print('genotype to neighbors', Counter([len(x) for x in genotype_to_neighbors]))

	# transition matrix
	# only allow one shift at a time
	shift_costs = [10]*4 + [200]*(2*(m-2))

	transitions = [[i] for i in range(p)]
	transition_costs = [[0] for i in range(p)]
	for i, state in enumerate(inheritance_states):
	    for j in range(len(shift_costs)):
	        for k in range(state[j]):
	            new_state = tuple(k if i == j else x for i, x in enumerate(state))
	            if new_state in state_to_index:
	                new_index = state_to_index[new_state]
	                transitions[i].append(new_index)
	                transition_costs[i].append(shift_costs[j])
	                transitions[new_index].append(i)
	                transition_costs[new_index].append(shift_costs[j])
	            
	transitions = np.array(transitions)
	transition_costs = np.array(transition_costs)
	print('transitions', transitions.shape)

	# loss matrix
	losses = np.zeros((p, q)) - 1
	for i, s in enumerate(inheritance_states):
	    
	    # what genotypes can be validly produced from this inheritance state?          
	    valid_genotypes = np.zeros((anc_variants.shape[0], m), dtype=int)
	        
	    # mom
	    if s[0] == 0 and s[1] == 0:
	        valid_genotypes[:, 0] = anc_variants[:, 0] + anc_variants[:, 1]
	    elif s[0] == 0:
	        valid_genotypes[:, 0] = 2*anc_variants[:, 0]
	    elif s[1] == 0:
	        valid_genotypes[:, 0] = 2*anc_variants[:, 1]
	    else:
	        valid_genotypes[:, 0] = -1
	            
	    # dad
	    if s[2] == 0 and s[3] == 0:
	        valid_genotypes[:, 1] = anc_variants[:, 2] + anc_variants[:, 3]
	    elif s[2] == 0:
	        valid_genotypes[:, 1] = 2*anc_variants[:, 2]
	    elif s[3] == 0:
	        valid_genotypes[:, 1] = 2*anc_variants[:, 3]
	    else:
	        valid_genotypes[:, 1] = -1
	        
	    # children
	    for index in range(m-2):
	        mat, pat = s[(4+(2*index)):(6+(2*index))]
	        
	        if s[mat] == 0 and s[2+pat] == 0:
	            valid_genotypes[:, 2+index] = anc_variants[:, mat] + anc_variants[:, 2+pat]
	        elif s[mat] == 0:
	            valid_genotypes[:, 2+index] = 2*anc_variants[:, mat]
	        elif s[2+pat] == 0:
	            valid_genotypes[:, 2+index] = 2*anc_variants[:, 2+pat]
	        else:
	            valid_genotypes[:, 2+index] = -1

	    valid_genotypes = set([genotype_to_index[tuple(x)] for x in valid_genotypes]) 

	    # breadth first search to fill in loss for this state
	    current_cost = 0
	    while len(valid_genotypes) > 0:
	        # add equivalents
	        valid_genotypes = valid_genotypes | set(chain.from_iterable([genotype_to_equivalents[x] for x in valid_genotypes]))

	        next_gen = set()
	        for g in valid_genotypes:
	            # fill in loss matrix
	            if losses[i, g] == -1:
	                losses[i, g] = current_cost

	                # pull next generation
	                next_gen.update([ng for ng in genotype_to_neighbors[g] if losses[i, ng] == -1])

	        valid_genotypes = next_gen
	        current_cost += 1  	    
	                
	# Check if we've missed some
	print('losses', losses.shape, 'missing', np.sum(losses == -1)/(losses.shape[0]*losses.shape[1]))

	families_of_this_size = [(fkey, ind_indices) for fkey, ind_indices in family_to_indices.items() if len(ind_indices) == m]
	for i, (fkey, ind_indices) in enumerate(families_of_this_size):
		family_index = family_to_index[fkey]
		print('family', fkey, family_index)

		# filter out all_hom_ref
		family_genotypes = whole_chrom[ind_indices, :].A
		family_indices = ~np.all(family_genotypes == 0, axis=0)
		family_genotypes = family_genotypes[:, family_indices]
		family_snp_positions = snp_positions[family_indices]
		m, n = family_genotypes.shape
		print('family chrom shape', m, n)

		# genotypes for each position
		pos_to_genindex = [genotype_to_index[tuple(x)] for x in family_genotypes.T]

		# viterbi
		v_cost = np.zeros((p, n+1), dtype=int)
		v_traceback = np.zeros((p, n+1), dtype=int)

		# forward sweep
		prev_time = time.time()
		v_traceback[:, 0] = -1
		for j in range(n): 
		    total_cost = v_cost[transitions, j] + transition_costs
		    min_index = np.argmin(total_cost, axis=1)
		    v_traceback[:, j+1] = transitions[range(p), min_index]
		    v_cost[:, j+1] = total_cost[range(p), min_index] + losses[:, pos_to_genindex[j]]
		print('Forward sweep complete', time.time()-prev_time, 'sec') 

		# backward sweep
		prev_time = time.time()
		final_states = np.zeros((n,), dtype=int)
		index = n
		k = np.argmin(v_cost[:, index])
		while index > 0:
		    final_states[index-1] = k    
		    k = v_traceback[k, index]
		    index -= 1
		print('Backward sweep complete', time.time()-prev_time, 'sec') 
		all_states[family_index, family_indices] = final_states
		all_losses[family_index, family_indices] = losses[final_states, pos_to_genindex]

	# save to file
	with open('phased/chr.%s.families.txt' % chrom, 'w+') as f:
		for fkey, index in sorted(family_to_index.items(), key=lambda x: x[1]):
			f.write('%s\t%s\t%s\t%d\n' % (fkey[0], fkey[1], fkey[2], len(families[fkey])))

	np.savez('phased/chr.%s.phased' % chrom, 
			states=all_states, losses=all_losses)

			
