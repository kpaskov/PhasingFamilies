import sys
from collections import Counter
from scipy.stats import binom
import numpy as np

family_sizes = [int(x) for x in sys.argv[1].split(',')]
phase_dir = sys.argv[2]
data_dir = sys.argv[3]
misc_dir = sys.argv[4]
#misc_dir = '../data'
#phase_dir = '../sherlock_phased_ssc'
#data_dir = '../split_gen_ssc'

for chrom in range(2, 23):
	chrom = str(chrom)
	print('Starting chromosome %s...' % chrom)

	# Pull chrom length from cytogenetic coordinates
	chrom_length = 0
	xticks, xticklabels = [], []
	with open('%s/cytoBand.txt' % misc_dir, 'r') as f:
	    for line in f:
	        pieces = line.strip().split()
	        if pieces[0] == 'chr%s' % chrom:
	            xticks.append(int(pieces[1]))
	            xticklabels.append(pieces[3])
	            chrom_length = max(chrom_length, int(pieces[1]), int(pieces[2]))
	print('Chrom length', chrom_length)
	    
	# pull positions that appear as deletion start/endpoints
	snp_positions = set([1, chrom_length])
	for j in family_sizes:
	    with open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, j), 'r')  as phasef:
	        next(phasef) # skip header
	        
	        for line in phasef:
	            pieces = line.strip().split('\t')
	            start_pos, end_pos = [int(x) for x in pieces[(1+(j*2)):(3+(j*2))]]
	            snp_positions.add(start_pos)
	            snp_positions.add(end_pos)
	            
	snp_positions = np.asarray(sorted(snp_positions))
	pos_to_index = dict([(x, i) for i, x in enumerate(snp_positions)])
	n = snp_positions.shape[0]
	print('n', n)

	# pull sample_ids
	sample_ids = []
	parent_ids = set()
	for j in family_sizes:
	    with open('%s/chr.%s.familysize.%d.families.txt' % (phase_dir, chrom, j), 'r')  as famf:
	        next(famf) # skip header
	        for line in famf:
	            individuals = line.strip().split('\t')[1:(1+j)]
	    
	            parent_ids.update(individuals[:2])
	            sample_ids.extend(individuals)
	            
	# Remove individuals from multiple families
	appearances = Counter(sample_ids)
	print('Removing individuals in multiple families', len([k for k, v in appearances.items() if v>1]))
	sample_ids = [x for x in sample_ids if appearances[x] == 1]

	# map sample ids to index
	sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])
	parent_indices = [i for i, x in enumerate(sample_ids) if x in parent_ids]
	child_indices = [i for i, x in enumerate(sample_ids) if x not in parent_ids]
	m = len(sample_id_to_index)
	print('m', m, 'parents', len(parent_indices), 'children', len(child_indices))

	# load deletions along with inherited/not inherited counts
	deletions = np.zeros((m, n), dtype=int)-1
	inherited = np.zeros((max(family_sizes)-1, n), dtype=int)
	not_inherited = np.zeros((max(family_sizes)-1, n), dtype=int)

	for j in family_sizes:
	    with open('%s/chr.%s.familysize.%d.families.txt' % (phase_dir, chrom, j), 'r')  as famf, \
	    open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, j), 'r')  as phasef:
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
	               
	            if fam_individuals[0] in sample_id_to_index:
	                deletions[sample_id_to_index[fam_individuals[0]], start_index:(end_index+1)] = sum(del_state[:2])
	            if fam_individuals[1] in sample_id_to_index:
	                deletions[sample_id_to_index[fam_individuals[1]], start_index:(end_index+1)] = sum(del_state[2:4])
	            
	            par_inh = [0, 0, 0, 0]
	            for k, child_id in enumerate(fam_individuals[2:]):
	                mat, pat = inheritance_state[(4+(2*k)):(6+(2*k))]
	                if mat is not None and pat is not None:
	                    if child_id in sample_id_to_index:
	                        deletions[sample_id_to_index[child_id], start_index:(end_index+1)] = (del_state[mat]+del_state[2+pat])
	                    par_inh[mat] += 1
	                    par_inh[2+pat] += 1
	                elif (mat is None) and (pat is not None) and (del_state[0] == del_state[1]):
	                    if child_id in sample_id_to_index:
	                        deletions[sample_id_to_index[child_id], start_index:(end_index+1)] = (del_state[0]+del_state[2+pat])
	                    par_inh[2+pat] += 1
	                elif (pat is None) and (mat is not None) and (del_state[2] == del_state[3]):
	                    if child_id in sample_id_to_index:
	                        deletions[sample_id_to_index[child_id], start_index:(end_index+1)] = (del_state[mat]+del_state[2])
	                    par_inh[mat] += 1
	                elif (del_state[0] == del_state[1]) and (del_state[2] == del_state[3]):
	                    if child_id in sample_id_to_index:
	                        deletions[sample_id_to_index[child_id], start_index:(end_index+1)] = (del_state[0]+del_state[2])
	                    
	            not_inherited[par_inh[0]+par_inh[1], start_index:(end_index+1)] += (int(par_inh[0]==0) + int(par_inh[1]==0))
	            inherited[par_inh[0]+par_inh[1], start_index:(end_index+1)] += (int(par_inh[0]!=0) + int(par_inh[1]!=0))
	            
	            not_inherited[par_inh[2]+par_inh[3], start_index:(end_index+1)] += (int(par_inh[2]==0) + int(par_inh[3]==0))
	            inherited[par_inh[2]+par_inh[3], start_index:(end_index+1)] += (int(par_inh[2]!=0) + int(par_inh[3]!=0))  
	            

	# fix up missing deletions where we can
	print('Percent missing', np.sum(deletions==-1)/(deletions.shape[0]*deletions.shape[1]))

	# extend first and last calls to end
	first_nonmissing = np.argmax(deletions != -1, axis=1)
	last_nonmissing = np.argmax(np.flip(deletions, axis=1) != -1, axis=1)
	for i, (fnm, lnm) in enumerate(zip(first_nonmissing, last_nonmissing)):
	    deletions[i, 0:fnm] = 0
	    deletions[i, (-1-lnm):n] = 0
	print('Inferred ends, Percent missing', np.sum(deletions==-1)/(deletions.shape[0]*deletions.shape[1]))

	# fill in intermediate missing values
	miss_start_x, miss_start_y = np.where((deletions[:, :-1] != -1) & (deletions[:, 1:] == -1))
	miss_end_x, miss_end_y = np.where((deletions[:, :-1] == -1) & (deletions[:, 1:] != -1))

	print('start and end xs should be equal', np.array_equal(miss_start_x, miss_end_x))
	for i, start, end in zip(miss_start_x, miss_start_y, miss_end_y):
	    if deletions[i, start] == deletions[i, end+1]:
	        deletions[i, (start+1):(end+1)] = deletions[i, start]
	print('Inferred intermediate missing, Percent missing', np.sum(deletions==-1)/(deletions.shape[0]*deletions.shape[1]))

	# Remove deletions in tricky regions
	pvalues = np.zeros((n,))
	for i in range(n):
	    for j in range(1, max(family_sizes)-1):
	        p_not_inherited = pow(0.5, j)
	        inh, notinh = inherited[j, i], not_inherited[j, i]
	        total = inh+notinh
	        if total > 0:
	            pvalues[i] = max(pvalues[i], -np.log10(binom.cdf(notinh, total, p_not_inherited)), -np.log10(binom.sf(notinh, total, p_not_inherited)))
	    
	cutoff = 2 + np.log10(n)
	print('removing %d indices due to imbalanced transmission' % np.sum(pvalues >= cutoff))
	print('%d single deletion positions removed' % np.sum(deletions[:, pvalues >= cutoff]==1))
	print('%d double deletion positions removed' % np.sum(deletions[:, pvalues >= cutoff]==2))

	deletions = deletions[:, pvalues < cutoff]
	snp_positions = snp_positions[pvalues < cutoff]
	print('Percent missing', np.sum(deletions==-1)/(deletions.shape[0]*deletions.shape[1]))

	# write to file
	np.savez_compressed('%s/chr.%s.deletions' % (phase_dir, chrom), 
	                    deletions=deletions, snp_positions=snp_positions, sample_ids=sample_ids)
