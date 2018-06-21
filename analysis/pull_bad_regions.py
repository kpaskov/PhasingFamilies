from collections import defaultdict
import numpy as np
from scipy.stats import chisquare


family_sizes = [3, 4, 5, 6, 7]
phase_dir = 'sherlock_phased'
data_dir = 'split_gen'

# sample_ids
sample_file = 'split_gen/chr.22.gen.samples.txt'
#sample_file = 'split_gen/chr.%s.gen.samples.txt' % ('X' if chrom.startswith('PAR') else chrom)
with open(sample_file, 'r') as f:
    sample_ids = [line.strip() for line in f]
sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])
m = max(sample_id_to_index.values())+1
print('m', m)

dels = []
snppos = []

for chrom in range(1, 23):
    chrom = str(chrom)
    print(chrom)

    # Pull centromere positions and chrom length
    chrom_length = 0
    centromere_start, centromere_end = None, None
    with open('cytoBand.txt', 'r') as f:
        xticks, xticklabels = [], []
        for line in f:
            pieces = line.strip().split()
            if pieces[0] == 'chr%s' % chrom:
                chrom_length = max(chrom_length, int(pieces[2]))
                
                if pieces[4] == 'acen':
                    if centromere_start is None:
                        centromere_start, centromere_end = int(pieces[1]), int(pieces[2])
                    else:
                        centromere_start, centromere_end = min(centromere_start, int(pieces[1])), max(centromere_end, int(pieces[2]))
    print('cent_start', centromere_start, 'cent_end', centromere_end, 'chrom_length', chrom_length)
    
    # snp positions
    clean_file = '%s/clean_indices_%s.txt' % (data_dir, 'X' if chrom.startswith('PAR') else chrom) 
    snp_positions = []
    with open(clean_file, 'r') as f:
        for i, line in enumerate(f):
            index, position = line.strip().split('\t')
            snp_positions.append(int(position))
    snp_positions = np.array(snp_positions)
    print('snp pos', snp_positions.shape)

    # pull positions that appear as deletion start/endpoints
    important_indices = set([0, len(snp_positions)-1])
    for j in family_sizes:
        with open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, j), 'r')  as phasef:
            next(phasef) # skip header

            for line in phasef:
                pieces = line.strip().split('\t')
                start_pos, end_pos, start_index, end_index = [int(x) for x in pieces[(1+(j*2)):(5+(j*2))]]
                important_indices.add(start_index)
                important_indices.add(end_index)

    important_indices = sorted(important_indices)
    snp_positions = snp_positions[important_indices]
    old_index_to_new_index = dict([(x, i) for i, x in enumerate(important_indices)])
    n = snp_positions.shape[0]
    print('n', n)

    deletions = np.zeros((m, n), dtype=int)-1
    inherited = np.zeros((max(family_sizes)-1, n), dtype=int)
    not_inherited = np.zeros((max(family_sizes)-1, n), dtype=int)

    # load deletions
    for j in family_sizes:
        with open('%s/chr.%s.familysize.%d.families.txt' % (phase_dir, chrom, j), 'r')  as famf, \
        open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, chrom, j), 'r')  as phasef:
            next(famf) # skip header
            next(phasef) # skip header

            fam_pieces = next(famf).strip().split('\t')
            fam_individuals = fam_pieces[1:(1+j)]
            fam_indices = [sample_id_to_index[ind] for ind in fam_individuals]

            for line in phasef:
                pieces = line.strip().split('\t')
                family_key = pieces[0]
                inheritance_state = [None if x == '*' else int(x) for x in pieces[1:(1+(j*2))]]
                del_state = [0 if x is None else x for x in inheritance_state[:4]]
                start_pos, end_pos, start_index, end_index = [int(x) for x in pieces[(1+(j*2)):(5+(j*2))]]
                start_index, end_index = old_index_to_new_index[start_index], old_index_to_new_index[end_index]

                # make sure we're on the right family
                while family_key != fam_pieces[0]:
                    fam_pieces = next(famf).strip().split('\t')
                    fam_individuals = fam_pieces[1:(1+j)]
                    fam_indices = [sample_id_to_index[ind] for ind in fam_individuals]

                deletions[fam_indices[0], start_index:(end_index+1)] = sum(del_state[:2])
                deletions[fam_indices[1], start_index:(end_index+1)] = sum(del_state[2:4])

                par_inh = [0, 0, 0, 0]
                for k, child_index in enumerate(fam_indices[2:]):
                    mat, pat = inheritance_state[(4+(2*k)):(6+(2*k))]
                    if mat is not None and pat is not None:
                        deletions[child_index, start_index:(end_index+1)] = (del_state[mat]+del_state[2+pat])
                        par_inh[mat] += 1
                        par_inh[2+pat] += 1
                    elif (mat is None) and (pat is not None) and (del_state[0] == del_state[1]):
                        deletions[child_index, start_index:(end_index+1)] = (del_state[0]+del_state[2+pat])
                        par_inh[2+pat] += 1
                    elif (pat is None) and (mat is not None) and (del_state[2] == del_state[3]):
                        deletions[child_index, start_index:(end_index+1)] = (del_state[mat]+del_state[2])
                        par_inh[mat] += 1
                    elif (del_state[0] == del_state[1]) and (del_state[2] == del_state[3]):
                        deletions[child_index, start_index:(end_index+1)] = (del_state[0]+del_state[2])

                not_inherited[par_inh[0]+par_inh[1], start_index:(end_index+1)] += (int(par_inh[0]==0) + int(par_inh[1]==0))
                inherited[par_inh[0]+par_inh[1], start_index:(end_index+1)] += (int(par_inh[0]!=0) + int(par_inh[1]!=0))

                not_inherited[par_inh[2]+par_inh[3], start_index:(end_index+1)] += (int(par_inh[2]==0) + int(par_inh[3]==0))
                inherited[par_inh[2]+par_inh[3], start_index:(end_index+1)] += (int(par_inh[2]!=0) + int(par_inh[3]!=0))  

    print('deletions loaded')
    
    # Remove deletions in tricky regions
    pvalues = np.ones((n,))
    for i in range(n):
        actual = []
        expected = []
        for j in range(1, inherited.shape[0]):
            p_not_inherited = pow(0.5, j)
            inh, notinh = inherited[j, i], not_inherited[j, i]
            total = inh+notinh
            if total != 0:
                actual.extend([inh, notinh])
                expected.extend([total*(1-p_not_inherited), total*p_not_inherited])

        pvalues[i] = chisquare(actual, expected)[1]
    pvalues[(snp_positions>=centromere_start) & (snp_positions<=centromere_end)] = 0
    print('pvalues calculated')
    
    region_switch = np.where((pvalues[:-1] <= 0.01/n) != (pvalues[1:] <= 0.01/n))[0]
    with open('%s/chr.%s.bad_regions.txt' % (phase_dir, chrom), 'w+') as f:
        if pvalues[0] <= 0.01/n:
            start = 1
        else:
            start = snp_positions[region_switch[0]]
            region_switch = region_switch[1:]
            
        for i in range(0, region_switch.shape[0]-1, 2):
            f.write('%d\t%d\n' % (start, snp_positions[region_switch[i]]))
            start = snp_positions[region_switch[i+1]]

        if pvalues[-1] <= 0.01/n:
            f.write('%d\t%d\n' % (start, chrom_length))

