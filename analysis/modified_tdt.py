import sys
import json
import numpy as np
import bayesian_transmission_rate as btr

deletion_dir = sys.argv[1] # 'deletions_ihart_asym'
ped_file = sys.argv[2] #'../datav34.vcf.ped'
chrom = sys.argv[3]

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

# pull deletions
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

def create_contingency(collections, family_sizes, filter_child):
	# collection, family_size, notrans/trans
	contingency = np.zeros((len(collections), len(family_sizes), 2))
	
	for j, c in enumerate(collections):
		for d in c['matches']:
			contingency[j, familysize_to_index[d['family_size']], 0] += len([x for x in d['notrans'] if filter_child(x, d['is_mat'])])
			contingency[j, familysize_to_index[d['family_size']], 1] += len([x for x in d['trans'] if filter_child(x, d['is_mat'])])
	return contingency

contingency_aff = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='2')
contingency_unaff = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='1')

contingency_aff_m = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='2' and child_id_to_sex[child_id]=='1')
contingency_unaff_m = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='1' and child_id_to_sex[child_id]=='1')

contingency_aff_f = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='2' and child_id_to_sex[child_id]=='2')
contingency_unaff_f = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='1' and child_id_to_sex[child_id]=='2')

contingency_aff_mat = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='2' and is_mat)
contingency_unaff_mat = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='1' and is_mat)

contingency_aff_pat = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='2' and not is_mat)
contingency_unaff_pat = create_contingency(collections, family_sizes, lambda child_id, is_mat: child_id_to_affected[child_id]=='1' and not is_mat)

np.savez('%s/chr.%s.contingency' % (deletion_dir, chrom), 
	aff=contingency_aff, unaff=contingency_unaff, 
	aff_m=contingency_aff_m, unaff_m=contingency_unaff_m, 
	aff_f=contingency_aff_f, unaff_f=contingency_unaff_f,
	aff_mat=contingency_aff_mat, unaff_mat=contingency_unaff_mat,
	aff_pat=contingency_aff_pat, unaff_pat=contingency_unaff_pat)
print('contingencies complete')

transrates_aff = btr.calculate_transmission_rates(contingency_aff, family_sizes)
transrates_unaff = btr.calculate_transmission_rates(contingency_unaff, family_sizes)

transrates_aff_m = btr.calculate_transmission_rates(contingency_aff_m, family_sizes)
transrates_unaff_m = btr.calculate_transmission_rates(contingency_unaff_m, family_sizes)

transrates_aff_f = btr.calculate_transmission_rates(contingency_aff_f, family_sizes)
transrates_unaff_f = btr.calculate_transmission_rates(contingency_unaff_f, family_sizes)

transrates_aff_mat = btr.calculate_transmission_rates(contingency_aff_mat, family_sizes)
transrates_unaff_mat = btr.calculate_transmission_rates(contingency_unaff_mat, family_sizes)

transrates_aff_pat = btr.calculate_transmission_rates(contingency_aff_pat, family_sizes)
transrates_unaff_pat = btr.calculate_transmission_rates(contingency_unaff_pat, family_sizes)

np.savez('%s/chr.%s.transrates' % (deletion_dir, chrom), 
	aff=transrates_aff, unaff=transrates_unaff, 
	aff_m=transrates_aff_m, unaff_m=transrates_unaff_m, 
	aff_f=transrates_aff_f, unaff_f=transrates_unaff_f,
	aff_mat=transrates_aff_mat, unaff_mat=transrates_unaff_mat,
	aff_pat=transrates_aff_pat, unaff_pat=transrates_unaff_pat)
print('transrates complete')

posteriors_aff = btr.calculate_posteriors(contingency_aff, family_sizes)
posteriors_unaff = btr.calculate_posteriors(contingency_unaff, family_sizes)

posteriors_aff_m = btr.calculate_posteriors(contingency_aff_m, family_sizes)
posteriors_unaff_m = btr.calculate_posteriors(contingency_unaff_m, family_sizes)

posteriors_aff_f = btr.calculate_posteriors(contingency_aff_f, family_sizes)
posteriors_unaff_f = btr.calculate_posteriors(contingency_unaff_f, family_sizes)

posteriors_aff_mat = btr.calculate_posteriors(contingency_aff_mat, family_sizes)
posteriors_unaff_mat = btr.calculate_posteriors(contingency_unaff_mat, family_sizes)

posteriors_aff_pat = btr.calculate_posteriors(contingency_aff_pat, family_sizes)
posteriors_unaff_pat = btr.calculate_posteriors(contingency_unaff_pat, family_sizes)

np.savez('%s/chr.%s.posteriors' % (deletion_dir, chrom), 
	aff=posteriors_aff, unaff=posteriors_unaff, 
	aff_m=posteriors_aff_m, unaff_m=posteriors_unaff_m, 
	aff_f=posteriors_aff_f, unaff_f=posteriors_unaff_f,
	aff_mat=posteriors_aff_mat, unaff_mat=posteriors_unaff_mat,
	aff_pat=posteriors_aff_pat, unaff_pat=posteriors_unaff_pat)
print('posteriors complete')

overlap = btr.calculate_posterior_overlap(posteriors_aff, posteriors_unaff)
overlap_m = btr.calculate_posterior_overlap(posteriors_aff_m, posteriors_unaff_m)
overlap_f = btr.calculate_posterior_overlap(posteriors_aff_f, posteriors_unaff_f)
overlap_mat = btr.calculate_posterior_overlap(posteriors_aff_mat, posteriors_unaff_mat)
overlap_pat = btr.calculate_posterior_overlap(posteriors_aff_pat, posteriors_unaff_pat)

np.savez('%s/chr.%s.posterior_overlaps' % (deletion_dir, chrom), 
	all=overlap, m=overlap_m, f=overlap_f,
	mat=overlap_mat, pat=overlap_pat)
print('posterior overlaps complete')

posterior_pvalues_aff = btr.calculate_posterior_pvalue(posteriors_aff)
posterior_pvalues_unaff = btr.calculate_posterior_pvalue(posteriors_unaff)

posterior_pvalues_aff_m = btr.calculate_posterior_pvalue(posteriors_aff_m)
posterior_pvalues_unaff_m = btr.calculate_posterior_pvalue(posteriors_unaff_m)

posterior_pvalues_aff_f = btr.calculate_posterior_pvalue(posteriors_aff_f)
posterior_pvalues_unaff_f = btr.calculate_posterior_pvalue(posteriors_unaff_f)

posterior_pvalues_aff_mat = btr.calculate_posterior_pvalue(posteriors_aff_mat)
posterior_pvalues_unaff_mat = btr.calculate_posterior_pvalue(posteriors_unaff_mat)

posterior_pvalues_aff_pat = btr.calculate_posterior_pvalue(posteriors_aff_pat)
posterior_pvalues_unaff_pat = btr.calculate_posterior_pvalue(posteriors_unaff_pat)

np.savez('%s/chr.%s.posterior_pvalues' % (deletion_dir, chrom), 
	aff=posterior_pvalues_aff, unaff=posterior_pvalues_unaff, 
	aff_m=posterior_pvalues_aff_m, unaff_m=posterior_pvalues_unaff_m, 
	aff_f=posterior_pvalues_aff_f, unaff_f=posterior_pvalues_unaff_f,
	aff_mat=posterior_pvalues_aff_mat, unaff_mat=posterior_pvalues_unaff_mat,
	aff_pat=posterior_pvalues_aff_pat, unaff_pat=posterior_pvalues_unaff_pat)
print('posterior pvalues complete')


