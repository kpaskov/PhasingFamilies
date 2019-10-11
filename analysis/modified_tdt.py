import sys
import json
import numpy as np
from collections import defaultdict
import scipy.stats as stats

deletion_dir = sys.argv[1] # 'deletions_ihart_asym'
ped_file = sys.argv[2] #'../datav34.vcf.ped'
chrom = sys.argv[3]

grid = np.arange(0.001, 1, 0.001)

# pull affected status
# (0=unknown; 1=unaffected; 2=affected)
child_id_to_affected = dict()
child_id_to_sex = dict()
with open(ped_file, 'r') as f:
	for line in f:
		pieces = line.strip().split('\t')
		if len(pieces) >= 6:
			fam_id, child_id, f_id, m_id, sex, disease_status = pieces[0:6]
			child_id_to_affected[child_id] = disease_status
			child_id_to_sex[child_id] = sex
print('ped loaded')
print('children', len(child_id_to_affected))

# pull collections
with open('%s/chr.%s.collections.json' % (deletion_dir, chrom), 'r') as f:
	collections = json.load(f)
print('collections loaded', len(collections))

# pull familysizes
family_sizes = [3, 4, 5, 6, 7]
familysize_to_index = dict([(x, i) for i, x in enumerate(family_sizes)])
print('family sizes', family_sizes)

# pull positions
positions = -np.ones((len(collections), 4), dtype=int)
positions[:, 0] = [c['deletion']['start_pos'] for c in collections]
positions[:, 1] = [c['deletion']['end_pos'] for c in collections]
positions[:, 2] = [c['deletion']['opt_start_pos'] for c in collections]
positions[:, 3] = [c['deletion']['opt_end_pos'] for c in collections]
np.save('%s/chr.%s.positions' % (deletion_dir, chrom), positions)
print('positions complete')

def remove_double_deletions(deletions):
    doublekey_to_deletions = defaultdict(list)
    for d in deletions:
        doublekey = (d['family'], d['is_mat'])
        doublekey_to_deletions[doublekey].append(d)
        
    return [ds[0] for ds in doublekey_to_deletions.values() if len(ds) == 1]

def create_transmission_table(deletions):
    # deletions must come from a family with at least two children
    deletions = [d for d in deletions if d['family_size']>3]

    if len(deletions) == 0:
    	return np.zeros((0, 2, 2), dtype=int), np.zeros((0,), dtype=int)
    
    # deletion, A/U, notrans/trans
    transmissions = np.zeros((len(deletions), 2, 2), dtype=int)
    for i, d in enumerate(deletions):
        transmissions[i, 0, 0] = len([x for x in d['notrans'] if child_id_to_affected[x] == '2'])
        transmissions[i, 0, 1] = len([x for x in d['trans'] if child_id_to_affected[x] == '2'])

        transmissions[i, 1, 0] = len([x for x in d['notrans'] if child_id_to_affected[x] == '1'])
        transmissions[i, 1, 1] = len([x for x in d['trans'] if child_id_to_affected[x] == '1'])

    vs, cs = np.unique(transmissions, axis=0, return_counts=True)
    return vs, cs

def create_transmission_table_mf(deletions):
    # deletions must come from a family with at least two children
    deletions = [d for d in deletions if d['family_size']>3]

    if len(deletions) == 0:
    	return np.zeros((0, 4, 2), dtype=int), np.zeros((0,), dtype=int)
    
    # deletion, AM/AF/UM/UF, notrans/trans
    transmissions = np.zeros((len(deletions), 4, 2), dtype=int)
    for i, d in enumerate(deletions):
        transmissions[i, 0, 0] = len([x for x in d['notrans'] if child_id_to_sex[x] == '1' and child_id_to_affected[x] == '2'])
        transmissions[i, 0, 1] = len([x for x in d['trans'] if child_id_to_sex[x] == '1' and child_id_to_affected[x] == '2'])

        transmissions[i, 1, 0] = len([x for x in d['notrans'] if child_id_to_sex[x] == '2' and child_id_to_affected[x] == '2'])
        transmissions[i, 1, 1] = len([x for x in d['trans'] if child_id_to_sex[x] == '2' and child_id_to_affected[x] == '2'])

        transmissions[i, 2, 0] = len([x for x in d['notrans'] if child_id_to_sex[x] == '1' and child_id_to_affected[x] == '1'])
        transmissions[i, 2, 1] = len([x for x in d['trans'] if child_id_to_sex[x] == '1' and child_id_to_affected[x] == '1'])

        transmissions[i, 3, 0] = len([x for x in d['notrans'] if child_id_to_sex[x] == '2' and child_id_to_affected[x] == '1'])
        transmissions[i, 3, 1] = len([x for x in d['trans'] if child_id_to_sex[x] == '2' and child_id_to_affected[x] == '1'])

    vs, cs = np.unique(transmissions, axis=0, return_counts=True)
    return vs, cs

def estimate_transmission(vs, counts, max_iters=5, l1_convergence_cutoff=np.power(10.0, -2)):
    prior = stats.beta.pdf(grid, 2, 2)

    prev_ps = None
    ps = np.array([0.5 for _ in range(vs.shape[1])])
    diff = 1
    num_iters = 0
    
    while diff>l1_convergence_cutoff and num_iters<max_iters:
        posteriors = np.array([np.log(prior) for _ in range(vs.shape[1])]).T
        
        # for each transmission probability
        for index in range(vs.shape[1]):

            for i in range(vs.shape[0]):
                ns = np.sum(vs[i, :, :], axis=1)

                p_nt = np.ones(grid.shape)
                for j, (p, n) in enumerate(zip(ps, ns)):
                    if j==index:
                        p_nt = p_nt * np.power(1-grid, n)
                    else:
                        p_nt = p_nt * np.power(1-p, n)

                for j in np.where(ns > 0)[0]:
                    if j==index:
                        qs = grid
                    else:
                        qs = ps[j]

                    posteriors[:, index] += counts[i] * ((vs[i, j, 1] * np.log(qs)) + (vs[i, j, 0] * np.log(1-qs)) - np.log(1-p_nt))

            # switch back to probability space
            posteriors[:, index] = np.exp(posteriors[:, index] - np.max(posteriors[:, index]))
            posteriors[:, index] = posteriors[:, index]/np.sum(posteriors[:, index])
         
        # check convergence
        ps = np.array([np.mean(grid.shape[0]*grid*posteriors[:, i]) for i in range(vs.shape[1])])
        diff = 1 if prev_ps is None else np.sum(np.abs(ps-prev_ps))
        prev_ps = ps
        num_iters += 1

    #if diff>l1_convergence_cutoff:
    #	print('did not converge')
    #	#posteriors[:] = np.nan
    #	#ps = [np.nan, np.nan, np.nan, np.nan]

    return posteriors, ps, diff

# collection, grid, A/U
all_posteriors = np.zeros((len(collections), grid.shape[0], 2))
# collection, A/U
all_transrates = np.zeros((len(collections), 2))

# collection, grid, AM/AF/UM/UF
all_posteriors_mf = np.zeros((len(collections), grid.shape[0], 4))
# collection, AM/AF/UM/UF
all_transrates_mf = np.zeros((len(collections), 4))

for i, collection in enumerate(collections):
	deletions = remove_double_deletions(collection['matches'])
	
	# all
	vs, cs = create_transmission_table(deletions)
	posteriors, trs, conf = estimate_transmission(vs, cs)
	all_posteriors[i, :, :] = posteriors
	all_transrates[i, :] = trs

	# m/f
	vs, cs = create_transmission_table_mf(deletions)
	posteriors, trs, conf = estimate_transmission(vs, cs)
	all_posteriors_mf[i, :, :] = posteriors
	all_transrates_mf[i, :] = trs

	if i%100 == 0:
		print(i, '/', len(collections))

np.savez('%s/chr.%s.transrates' % (deletion_dir, chrom), 
	aff=all_transrates[:, 0], unaff=all_transrates[:, 1],
	aff_m=all_transrates_mf[:, 0], unaff_m=all_transrates_mf[:, 2], 
	aff_f=all_transrates_mf[:, 1], unaff_f=all_transrates_mf[:, 3])
print('transrates complete')

np.savez('%s/chr.%s.posteriors' % (deletion_dir, chrom), 
	aff=all_posteriors[:, :, 0], unaff=all_posteriors[:, 1],
	aff_m=all_posteriors_mf[:, :, 0], unaff_m=all_posteriors_mf[:, :, 2], 
	aff_f=all_posteriors_mf[:, :, 1], unaff_f=all_posteriors_mf[:, :, 3])
print('posteriors complete')

# collection
overlaps = np.zeros((len(collections),))

# collection, M/F
overlaps_mf = np.zeros((len(collections), 2))

for i in range(len(collections)):
	overlaps[i] = np.sum(np.minimum(all_posteriors[i, :, 0], all_posteriors[i, :, 1]))/np.sum(all_posteriors[i, :, 0])
	overlaps_mf[i, 0] = np.sum(np.minimum(all_posteriors_mf[i, :, 0], all_posteriors_mf[i, :, 2]))/np.sum(all_posteriors_mf[i, :, 0])
	overlaps_mf[i, 1] = np.sum(np.minimum(all_posteriors_mf[i, :, 1], all_posteriors_mf[i, :, 3]))/np.sum(all_posteriors_mf[i, :, 1])
	
np.savez('%s/chr.%s.posterior_overlaps' % (deletion_dir, chrom), 
	all=overlaps,
	m=overlaps_mf[:, 0], f=overlaps_mf[:, 1])
print('overlap_pvalues complete')


# collection, A/U, higher/lower
posterior_pvalues = np.zeros((len(collections), 2, 2))

# collection, AM/AF/UM/UF, higher/lower
posterior_pvalues_mf = np.zeros((len(collections), 4, 2))

for j in range(2):
	posterior_pvalues[:, j, 0] = np.sum(all_posteriors[:, grid<=0.5, j], axis=1)/np.sum(all_posteriors[:, :, j], axis=1)
	posterior_pvalues[:, j, 1] = np.sum(all_posteriors[:, grid>=0.5, j], axis=1)/np.sum(all_posteriors[:, :, j], axis=1)

for j in range(4):
	posterior_pvalues_mf[:, j, 0] = np.sum(all_posteriors_mf[:, grid<=0.5, j], axis=1)/np.sum(all_posteriors_mf[:, :, j], axis=1)
	posterior_pvalues_mf[:, j, 1] = np.sum(all_posteriors_mf[:, grid>=0.5, j], axis=1)/np.sum(all_posteriors_mf[:, :, j], axis=1)


np.savez('%s/chr.%s.posterior_pvalues' % (deletion_dir, chrom), 
	aff=posterior_pvalues[:, 0, :], unaff=posterior_pvalues[:, 1, :],
	aff_m=posterior_pvalues_mf[:, 0, :], unaff_m=posterior_pvalues_mf[:, 2, :], 
	aff_f=posterior_pvalues_mf[:, 1, :], unaff_f=posterior_pvalues_mf[:, 3, :])
print('posterior_pvalues complete')


