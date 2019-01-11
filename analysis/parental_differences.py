import numpy as np
from scipy import sparse
from os import listdir
from scipy.stats import chi2_contingency, fisher_exact
import sys

chrom = sys.argv[1]
data_dir = sys.argv[2]
ped_file = sys.argv[3]
out_dir = sys.argv[4]

bonferonni_cutoff = 11


# load data for parents
mom_indices, dad_indices = set(), set()

with open('%s/chr.%s.gen.samples.txt' % (data_dir, chrom), 'r') as f:
    sample_id_to_index = dict([(x.strip(), i) for i, x in enumerate(f)])
    
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) >= 5:
            if pieces[5] == '2':
                if pieces[3] in sample_id_to_index:
                    mom_indices.add(sample_id_to_index[pieces[3]])
                if pieces[2] in sample_id_to_index:
                    dad_indices.add(sample_id_to_index[pieces[2]])

print('Moms', len(mom_indices), 'Dads', len(dad_indices), 'Overlap', len(mom_indices & dad_indices))

# pull snp positions
pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))
snp_positions = pos_data[:, 1]
is_snp = pos_data[:, 2].astype(bool)

# pull genotype data from .npz
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])
whole_chrom = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

# filter only monoallelic snps
whole_chrom = whole_chrom[:, is_snp]
snp_positions = snp_positions[is_snp]
whole_chrom[whole_chrom<-1] = -1
m, n = whole_chrom.shape

print('chrom shape only SNPs', m, n)

mom_gen = whole_chrom[list(mom_indices), :].A
dad_gen = whole_chrom[list(dad_indices), :].A

num_moms, num_dads = len(mom_indices), len(dad_indices)
cached_logpvalues = -np.ones((num_moms+1, num_dads+1), dtype=int)

def calc_pvalue(m, d):
    if cached_logpvalues[m, d] == -1:
        dc = np.asarray([[m, d],
                         [num_moms-m, num_dads-d]])
        
        if np.all(dc>5):
            pvalue = chi2_contingency(dc, correction=True)[1]
            if pvalue == 0:
                cached_logpvalues[m, d] = 1000
            else:
                cached_logpvalues[m, d] = -np.log10(pvalue)
        elif (m==0 and d==0) or (m==num_moms and d==num_dads):
            cached_logpvalues[m, d] = 0
        else:
            pvalue = fisher_exact(dc)[1]
            if pvalue == 0:
                cached_logpvalues[m, d] = 1000
            else:
                cached_logpvalues[m, d] = -np.log10(pvalue)
    return cached_logpvalues[m, d]

gens = [-1, 0, 1, 2]
par_gen_counts = np.zeros((n, len(gens), 2), dtype=int)
for i, gen in enumerate(gens):
    par_gen_counts[:, i, 0] = np.sum(mom_gen==gen, axis=0)
    par_gen_counts[:, i, 1] = np.sum(dad_gen==gen, axis=0)

log_pvalues = np.zeros((n, 4), dtype=int)
for i in range(n):
    for j, gen in enumerate(gens):
        log_pvalues[i, j] = calc_pvalue(par_gen_counts[i, j, 0], par_gen_counts[i, j, 1])

    if i % 100000 == 0:
        print(i)

np.save('%s/chr.%s.logpvalues' % (out_dir, chrom), log_pvalues)

# save interesting sites to txt
indices_of_interest = np.where(np.any(log_pvalues > bonferonni_cutoff, axis=1))[0]
snps_of_interest = snp_positions[indices_of_interest]
print(snps_of_interest.shape)

np.savetxt('%s/chr.%s.diffsites.txt' % (out_dir, chrom), 
	np.hstack((snps_of_interest[:, np.newaxis], log_pvalues[indices_of_interest, :],
		par_gen_counts[indices_of_interest, :, 0], 
		par_gen_counts[indices_of_interest, :, 1])), 
	header='\t'.join(['position', '-log_pvalue_miss', '-log_pvalue_homref', '-logpvalue_het', '-logpvalue_homalt',
		'mat_miss', 'mat_homref', 'mat_het', 'mat_homalt', 
		'pat_miss', 'pat_homref', 'pat_het', 'pat_homalt']),
	fmt='%d',
	delimiter='\t')


