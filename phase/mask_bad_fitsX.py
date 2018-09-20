import numpy as np
from itertools import chain, product
from collections import Counter
import random
from scipy import sparse
from os import listdir
import sys

# Run locally with python phase/mask_bad_fitsX.py 3 split_gen_miss phased

chrom = 'X'
m = int(sys.argv[1])
data_dir = sys.argv[2]
phase_dir = sys.argv[3]

error_rate = .02
smooth = 5000

# inheritance states
#
#
# for parents:
# (0, 0) -> double deletion
# (0, 1) -> deletion on parental1
# (1, 0) -> deletion on parental2
# (1, 1) -> normal
# 
# for children:
# (0, 0) -> m1p1
# (0, 1) -> m1p2
# (1, 0) -> m2p1
# (1, 1) -> m2p2
#
# for family:
# (0) -> can't model
# (1) -> we're good

if m >= 5:
	inheritance_states = np.array(list(product(*([[0, 1]]*(2*m)))), dtype=np.int8)
else:
	inheritance_states = np.array([x for x in product(*([[0, 1]]*(2*m))) if x[4]==0], dtype=np.int8)
state_to_index = dict([(tuple(x), i) for i, x in enumerate(inheritance_states)])
p, state_len = inheritance_states.shape
print('inheritance states', inheritance_states.shape)

# genotype (pred, obs): cost
g_cost = {
	(-1, 0): 1,
	(-1, 1): 1,
	(-1, 2): 1,
	(0, 0): 0,
	(0, 1): 1,
	(0, 2): 2,
	(1, 0): 1,
	(1, 1): 0,
	(1, 2): 1,
	(2, 0): 2,
	(2, 1): 1,
	(2, 2): 0
}


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

genotypes = np.array(list(product(*[[0, 1, 2]]*m)), dtype=np.int8)
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

# pull families
fam_to_individuals = dict()
with open(phase_dir + '/chr.%s.familysize.%d.families.txt' % (chrom, m), 'r') as f:
	next(f) # skip header
	for line in f:
		pieces = line.strip().split()
		fam_to_individuals[pieces[0]] = pieces[1:]

# pull families with sequence data
sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
with open(sample_file, 'r') as f:
	sample_ids = [line.strip() for line in f]
sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])

# use only "cleaned" variants - must be SNPs
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s' % chrom) in f and 'gen.npz' in f])

coordinates = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir,  chrom))
snp_positions = coordinates[:, 1]
snp_indices = coordinates[:, 2]==1

snp_positions = snp_positions[snp_indices]

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

with open('%s/chr.%s.familysize.%s.phased.masked.txt' % (phase_dir, chrom, m), 'w+') as statef:
	# write headers
	statef.write('\t'.join(['family_id', 'state_id', 'm1_state', 'm2_state', 'p1_state', 'p2_state',
			'\t'.join(['child%d_%s_state' % ((i+1), c) for i, c in product(range(m-2), ['m', 'p'])]),
			'start_pos', 'end_pos', 'start_family_index', 'end_family_index' 'pos_length', 'family_index_length']) + '\n')

	for fkey, inds in fam_to_individuals.items(): 
		print('family', fkey)
		
		ind_indices = [sample_id_to_index[x] for x in inds]

		#load from npz
		data = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file))[ind_indices, :] for gen_file in gen_files]).A
		data = data[:, snp_indices]

		n = 2*snp_positions.shape[0]+1
		family_genotypes = np.zeros((m, n), dtype=np.int8)
		family_genotypes[:, np.arange(1, n-1, 2)] = data
		family_genotypes[:, -2] = family_genotypes[:, -1]

		# if any family member is missing, set whole family to 0 - this has the effect of ignoring missing positions
		family_genotypes[:, np.any(family_genotypes<0, axis=0)] = 0

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
	    
		# load deletions for this family
		final_states = -np.ones(((m*2), n), dtype=np.int8)
		with open(phase_dir + '/chr.%s.familysize.%d.phased.txt' % (chrom, m), 'r') as f:
			for line in f:
				pieces = line.strip().split('\t')
				if pieces[0] == fkey:
					inheritance_state = [int(x) for x in pieces[1:(1+(m*2))]]
					start_pos, end_pos = [int(x) for x in pieces[(1+(m*2)):(3+(m*2))]]
					start_index, end_index = [int(x) for x in pieces[(3+(m*2)):(5+(m*2))]]
					final_states[:, start_index:(end_index+1)] = np.asarray(inheritance_state)[:, np.newaxis]

 
		fit = -np.ones((n,), dtype=int)
		prev_state = None
		prev_state_indices = None
		for j in range(n): 
			pos_gen = tuple(family_genotypes[:, j])
			loss = calculate_loss(pos_gen).astype(int)
			current_state = tuple(final_states[:, j])

			if current_state != prev_state:
				prev_state = current_state
				num_unknowns = len([x for x in current_state if x == -1])
				if num_unknowns>0:
					prev_state_indices = []
					for poss_itr in [iter(x) for x in product(*([[0, 1]]*num_unknowns))]:
						poss_state = tuple([x if x != -1 else next(poss_itr) for x in current_state])
						prev_state_indices.append(state_to_index[poss_state])
				else:
					prev_state_indices = [state_to_index[tuple(final_states[:, j])]]

			fit[j] = mult_factor[j]*np.min(loss[prev_state_indices])


		c = np.convolve(fit/m, np.ones(smooth,), mode='same')
		masked = (c>(error_rate*smooth)).astype(np.int8)
		print('Percent masked', 100*np.sum(masked)/n)
		final_states = np.append(final_states, masked[np.newaxis, :], axis=0)

		# write to file
		change_indices = [-1] + np.where(np.any(final_states[:, 1:]!=final_states[:, :-1], axis=0))[0].tolist()
		for j in range(1, len(change_indices)):
			s_start, s_end = change_indices[j-1]+1, change_indices[j]
			#assert np.all(final_states[:, s_start] == final_states[:, s_end])
			statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
							fkey, 
							'\t'.join(map(str, final_states[:, s_start])), 
							family_snp_positions[s_start, 0]+1, family_snp_positions[s_end, 1],
							s_start, s_end, 
							family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0], 
							s_end-s_start+1))

		# last entry
		s_start, s_end = change_indices[-1]+1, family_snp_positions.shape[0]-1
		#assert np.all(final_states[:, s_start] == final_states[:, s_end])
		statef.write('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n' % (
						fkey, 
						'\t'.join(map(str, final_states[:, s_start])), 
						family_snp_positions[s_start, 0]+1, family_snp_positions[s_end, 1],
						s_start, s_end, 
						family_snp_positions[s_end, 1]-family_snp_positions[s_start, 0]+1, 
						s_end-s_start+1))
		statef.flush()	




