import sys
from os import listdir

import gzip

import numpy as np
from scipy import sparse
from scipy.stats import chi2_contingency

# run with python phase/select_variants.py X data/160826.ped split_gen

chrom = sys.argv[1]
ped_file = sys.argv[2]
data_dir = sys.argv[3]

variant_file = '%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom)

# pull sample indices
sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
with open(sample_file, 'r') as f:
    individual_to_index = dict([(x.strip(), i) for i, x in enumerate(f)])

# pull mothers and fathers
mother_ids, father_ids = [], []
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) >= 6:
            fam_id, child_id, f_id, m_id, sex, disease_status = pieces[0:6]
            mother_ids.append(m_id)
            father_ids.append(f_id)
            
father_indices = [i for x, i in individual_to_index.items() if x in father_ids]
mother_indices = [i for x, i in individual_to_index.items() if x in mother_ids]
print('Fathers', len(father_indices), 'Mothers', len(mother_indices))
	
# pull snp indices and positions
snp_indices = []
snp_positions = []
with gzip.open(variant_file, 'rt') as f:
    for i, line in enumerate(f):
        pieces = line.strip().split('\t')
        if len(pieces[3]) == 1 and len(pieces[4]) == 1 and pieces[3] != '.' and pieces[4] != '.':
            snp_indices.append(i)
        snp_positions.append(int(pieces[1]))
snp_positions = np.array(snp_positions)

# remove positions with mother/father missingness bias
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s' % ('X' if chrom.startswith('PAR') else chrom)) in f and 'gen.npz' in f])

offset = 0
indices_with_bias = set()
miss_to_pvalue = dict()
for gen_file in gen_files:
	subchrom = sparse.load_npz('%s/%s' % (data_dir, gen_file))
	m, n = subchrom.shape
	
	father_missing = np.ravel((subchrom[father_indices, :]==-1).sum(axis=0).A)
	mother_missing = np.ravel((subchrom[mother_indices, :]==-1).sum(axis=0).A)
	
	has_missing = np.where(((father_missing+mother_missing) > 0) & (((father_missing+mother_missing) < (len(father_indices)+len(mother_indices)))))[0]
	for i in has_missing:
		fmis, mmis = father_missing[i], mother_missing[i]
		if (fmis, mmis) not in miss_to_pvalue:
			cont = [[fmis, mmis], [len(father_indices)-fmis, len(mother_indices)-mmis]]
			try:
				miss_to_pvalue[(fmis, mmis)] = chi2_contingency(cont)[1]
			except:
				print('Error', cont)
		pvalue = miss_to_pvalue[(fmis, mmis)]

		if pvalue <= pow(10, -7):
			indices_with_bias.add(i+offset)
	offset += n

num_clean_indices = 0
with open('%s/clean_indices_%s.txt' % (data_dir, chrom), 'w+') as f:
	for snp_index in snp_indices:
		if snp_index not in indices_with_bias:
			f.write('%d\t%d\n' % (snp_index, snp_positions[snp_index]))
			num_clean_indices += 1

print('Num clean indices', num_clean_indices)
print('Discarded %d nonSNPs and %d sex-biased' % (snp_positions.shape[0]-len(snp_indices), len(indices_with_bias)))
