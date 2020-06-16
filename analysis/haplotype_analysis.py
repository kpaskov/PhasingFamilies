
import scipy.sparse as sparse
import importlib.util
import numpy as np
import json
from os import listdir
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import scipy.stats
import gzip

## must be phased in quads for now
#data_dir = 'split_gen_spark'
#chrom = '10'
#interval_start_pos, interval_end_pos = 125000000, 126500000
#ped_file = 'data/spark.ped.quads.ped'
#param_file = 'parameter_estimation/params/spark_multiloss_params.json'
#phase_dir = 'phased_spark_quads'
#num_loss_regions = 1
#of_interest = 'data/10interest_38.bed'

data_dir = 'split_gen_ssc'
chrom = '10'
interval_start_pos, interval_end_pos = 126688569, 128188569
ped_file = 'data/ssc.ped'
param_file = 'parameter_estimation/params/ssc_multiloss_params.json'
phase_dir = 'phased_ssc'
num_loss_regions = 2
of_interest = 'data/10interest_37.bed'


# pull phasing code
spec = importlib.util.spec_from_file_location("input_output", "phase/input_output.py")
input_output = importlib.util.module_from_spec(spec)
spec.loader.exec_module(input_output)

spec = importlib.util.spec_from_file_location("ancestral_variants", "phase/ancestral_variants.py")
ancestral_variants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ancestral_variants)

spec = importlib.util.spec_from_file_location("inheritance_states", "phase/inheritance_states.py")
inheritance_states = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inheritance_states)

spec = importlib.util.spec_from_file_location("genotypes", "phase/genotypes.py")
genotypes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(genotypes)

spec = importlib.util.spec_from_file_location("losses", "phase/losses.py")
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


# pull genotype data
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])
gen = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

coordinates = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))
snp_positions = coordinates[:, 1]
is_snp = coordinates[:, 2]==1
is_pass = coordinates[:, 3]==1

pos_of_interest = []
with open(of_interest, 'r') as f:
	for line in f:
		pos = int(line.strip().split('-')[-1])
		pos_of_interest.append(pos)

indices = np.where(is_snp & is_pass & np.isin(snp_positions, pos_of_interest))[0].tolist()
new_snp_positions = np.array(pos_of_interest)
refs = ['N']*len(new_snp_positions)
alts = ['N']*len(new_snp_positions)
with gzip.open('%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom), 'rt') as f:
	for i, line in enumerate(f):
		pieces = line.strip().split('\t')
		if i in indices:
			assert new_snp_positions[indices.index(i)] == int(pieces[1])
			refs[indices.index(i)] = pieces[3]
			alts[indices.index(i)] = pieces[4]


new_gen = -np.ones((gen.shape[0], new_snp_positions.shape[0]), dtype=int)
new_gen[:, np.isin(pos_of_interest, snp_positions)] = gen[:, indices].A
gen = new_gen
snp_positions = new_snp_positions

with open('data/refs.txt', 'r') as reff, open('data/alts.txt', 'r') as altf:
	for ref, alt, real_ref, real_alt in zip(refs, alts, [x.strip() for x in reff], [x.strip() for x in altf]):
		if real_ref != ref or real_alt != alt:
			print(ref, alt, real_ref, real_alt)

#with open('data/refs.txt', 'w+') as f:
#	for ref in refs:
#		f.write(ref + '\n')
#with open('data/alts.txt', 'w+') as f:
#	for alt in alts:
#		f.write(alt + '\n')

print(gen.shape, snp_positions.shape)

with open('%s/chr.%s.gen.samples.txt' % (data_dir, chrom), 'r') as f:
    sample_id_to_index = dict([(x.strip(), i) for i, x in enumerate(f)])

# now we find ancestral variants
with open(param_file, 'r') as f:
    params = json.load(f)

families = input_output.pull_families('%s/chr.%s.gen.samples.txt' % (data_dir, chrom), ped_file)

family_to_index = dict([(x.id, i) for i, x in enumerate(families)])
# family, state, 0/1
family_to_states = -np.ones((len(families), 13, 4))
with open('%s/chr.%s.familysize.4.phased.txt' % (phase_dir, chrom), 'r') as f:
    next(f) # skip header
    for line in f:
        pieces = line.strip().split('\t')
        start_pos, end_pos = int(pieces[14]), int(pieces[15])
        weight = np.clip(min(end_pos, interval_end_pos) - max(start_pos, interval_start_pos), 0, None)
        if weight>0:
            state = np.array([int(x) for x in pieces[1:14]])
            key = '.'.join(pieces[0].split('.')[:3])
            if key not in family_to_index:
                key = pieces[0].split('.')[0]

            indices = np.where(state>=0)[0]
            family_to_states[family_to_index[key], indices, state[indices]] += weight

family_to_state = -np.ones((len(families), 13), dtype=int)
for value in range(4):
	family_to_state[family_to_states[:, :, value]/np.sum(family_to_states, axis=2) > 0.9] = value
family_to_state[:, -1] = num_loss_regions-1

gens = genotypes.Genotypes(4)
states = inheritance_states.InheritanceStates(families[0], True, True, num_loss_regions)
loss = losses.LazyLoss(states, gens, families[0], params, num_loss_regions)

ancs = -np.ones((len(families), 4, snp_positions.shape[0]))
for family_index, family in enumerate(families):
    family_gen = gen[[sample_id_to_index[x] for x in family.individuals], :]
    if np.all(family_to_state[family_index, :]>=0):
        state = states.get_original_state(family_to_state[family_index, :])
        variants = ancestral_variants.AncestralVariants(states, gens, loss, family, state)
        for i in range(snp_positions.shape[0]):
            if np.all(family_gen[:, i]>=0):
                ancs[family_index, :, i] = variants(tuple(family_gen[:, i]))
                
    if family_index%1000==0:
        print(family_index)
        
families_of_interest = np.any(ancs>=0, axis=(1, 2))
families = [families[i] for i in np.where(families_of_interest)[0]]
ancs = ancs[families_of_interest, :, :]
family_to_state = family_to_state[families_of_interest, :]
print(ancs.shape, family_to_state.shape, len(families))
all_ancestors = np.vstack(ancs)

# fill in missing values with mean
var_means = np.zeros((all_ancestors.shape[1]))
for i in range(all_ancestors.shape[1]):
    if np.sum(all_ancestors[:, i]>=0) > 0:
        m = np.sum(all_ancestors[:, i]==1)/np.sum(all_ancestors[:, i]>=0)
    else:
        m = 0
    var_means[i] = m
#np.save('data/var_means', var_means)
var_means = np.load('data/var_means.npy')

for i in range(all_ancestors.shape[1]):
    all_ancestors[all_ancestors[:, i] == -1, i] = m
print(all_ancestors.shape)

assert np.all(all_ancestors>=0) and np.all(all_ancestors<= 1)

# cluster to find haplotypes
n_clusters=2
kmeans = KMeans(n_clusters=n_clusters)
#cluster_labels = kmeans.fit_predict(all_ancestors)

kmeans.cluster_centers_ = np.load('data/cluster_centers.npy')
cluster_labels = kmeans.predict(all_ancestors)
silhouette_avg = silhouette_score(all_ancestors, cluster_labels)
#np.save('data/cluster_centers', kmeans.cluster_centers_)

print('silhouette', silhouette_avg)

sample_id_to_affected = dict()
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) >= 5:
            fam_id, child_id, f_id, m_id, sex, phen = pieces[:6]
            sample_id_to_affected[child_id] = phen


# now TDT

# haplotype, mat/pat, unaff/aff, notrans/trans
contingency = np.zeros((n_clusters, 2, 2, 2))
couples = []
for i, (family, state) in enumerate(zip(families, family_to_state)):
    m1, m2, p1, p2 = cluster_labels[(4*i):(4*(i+1))]
    children_aff = [int(sample_id_to_affected[x]=='2') for x in family.individuals[2:]]
    mat_phase_indices = [8, 10]
    pat_phase_indices = [9, 11]
    if m1 != m2:
        contingency[m1, 0, children_aff, (state[mat_phase_indices]==0).astype(int)] += 1
        contingency[m2, 0, children_aff, (state[mat_phase_indices]==1).astype(int)] += 1
    
    if p1 != p2:
        contingency[p1, 1, children_aff, (state[pat_phase_indices]==2).astype(int)] += 1
        contingency[p2, 1, children_aff, (state[pat_phase_indices]==3).astype(int)] += 1
    couples.append((m1, m2, p1, p2))
    
print(Counter(couples))

# unaff/aff, all/mat/pat
pvalues = np.ones((n_clusters, 2, 3))
for i in range(contingency.shape[0]):
    pvalues[i, 0, 0] = scipy.stats.binom_test(np.sum(contingency[i, :, 0, 1]), 
                                        np.sum(contingency[i, :, 0, :]), p=0.5, alternative='greater')
    pvalues[i, 1, 0] = scipy.stats.binom_test(np.sum(contingency[i, :, 1, 1]), 
                                        np.sum(contingency[i, :, 1, :]), p=0.5, alternative='greater')
    
    pvalues[i, 0, 1] = scipy.stats.binom_test(contingency[i, 0, 0, 1], 
                                        np.sum(contingency[i, 0, 0, :]), p=0.5, alternative='greater')
    pvalues[i, 1, 1] = scipy.stats.binom_test(contingency[i, 0, 1, 1], 
                                        np.sum(contingency[i, 0, 1, :]), p=0.5, alternative='greater')
    
    pvalues[i, 0, 2] = scipy.stats.binom_test(contingency[i, 1, 0, 1],
                                        np.sum(contingency[i, 1, 0, :]), p=0.5, alternative='greater')
    pvalues[i, 1, 1] = scipy.stats.binom_test(contingency[i, 1, 1, 1], 
                                        np.sum(contingency[i, 1, 1, :]), p=0.5, alternative='greater')

# haplotype, mat/pat, unaff/aff, notrans/trans
print('unaff')
print(contingency[0, :, 0, :])
print(scipy.stats.chi2_contingency(contingency[0, :, 0, :]))
# we see more maternal transmissions of hap1 and paternal transmissions of hap2 to unaffected children

print('aff')
print(contingency[0, :, 1, :])
print(scipy.stats.chi2_contingency(contingency[0, :, 1, :]))
# transmissions to affected children are normal

print('mat')
print(contingency[0, 0, :, :])
print(scipy.stats.chi2_contingency(contingency[0, 0, :, :]))

print('pat')
print(contingency[0, 1, :, :])
print(scipy.stats.chi2_contingency(contingency[0, 1, :, :]))


for i in range(n_clusters):
    print(i)
    print('aff', scipy.stats.binom_test(np.sum(contingency[i, :, 1, 1]), 
                                        np.sum(contingency[i, :, 1, :]), p=0.5, alternative='greater'))
    print('unaff', scipy.stats.binom_test(np.sum(contingency[i, :, 0, 1]), 
                                        np.sum(contingency[i, :, 0, :]), p=0.5, alternative='greater'))
    
    print('aff mat', scipy.stats.binom_test(contingency[i, 0, 1, 1], 
                                        np.sum(contingency[i, 0, 1, :]), p=0.5, alternative='greater'))
    print('unaff mat', scipy.stats.binom_test(contingency[i, 0, 0, 1], 
                                        np.sum(contingency[i, 0, 0, :]), p=0.5, alternative='greater'))
    print(contingency[i, 0, 0, 0], contingency[i, 0, 0, 1])
    
    print('aff pat', scipy.stats.binom_test(contingency[i, 1, 1, 1], 
                                        np.sum(contingency[i, 1, 1, :]), p=0.5, alternative='greater'))
    print('unaff pat', scipy.stats.binom_test(contingency[i, 1, 0, 1], 
                                        np.sum(contingency[i, 1, 0, :]), p=0.5, alternative='greater'))
    print(contingency[i, 1, 0, 0], contingency[i, 1, 0, 1])
          
print('hap1')
for i in np.where(np.sum(all_ancestors[cluster_labels==0, :], axis=0)/np.sum(cluster_labels==0)>0.5)[0]:
    print(snp_positions[i])
print('hap2')
for i in np.where(np.sum(all_ancestors[cluster_labels==1, :], axis=0)/np.sum(cluster_labels==1)>0.5)[0]:
    print(snp_positions[i])



