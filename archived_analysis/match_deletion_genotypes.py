import numpy as np
import scipy.sparse as sparse
from os import listdir
import matplotlib.pyplot as plt
import sys
import json

deletion_dir = sys.argv[1]
data_dir = sys.argv[2]
chrom = sys.argv[3]

# read samples
sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
with open(sample_file, 'r') as f:
	sample_id_to_index = dict([(line.strip(), i) for i, line in enumerate(f)])
    
# load genotypes
gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])

# pull snp positions
pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))
is_snp = pos_data[:, 2].astype(bool)
is_pass = pos_data[:, 3].astype(bool)
snp_positions = pos_data[:, 1]
print('Sites pulled from vcf:', snp_positions.shape[0])

# Pull data together
A = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

# only look at snps that PASS GATK filter
A = A[:, is_snp & is_pass]
snp_positions = snp_positions[is_snp & is_pass]
print('Removed %d sites that are not bi-allelic SNPs' % np.sum(~is_snp))
print('Removed %d sites that do not pass GATK' % np.sum(is_snp & ~is_pass))

# read in deletions
with open('%s/chr.%s.deletions.json' % (deletion_dir, chrom), 'r') as f:
	deletions = json.load(f)

for d in deletions:
	sample_indices = [sample_id_to_index[d['mother']], sample_id_to_index[d['father']]] + [sample_id_to_index[x] for x in d['trans']] + [sample_id_to_index[x] for x in d['notrans']]
	variant_indices = np.where((snp_positions >= d['start_pos']) & (snp_positions <= d['end_pos']))[0]
	#d['genotypes'] = int((A[np.ix_(sample_indices, variant_indices)] > 0).sum())
	d['genotypes'] = int(A[np.ix_(sample_indices, variant_indices)].nnz)

with open('%s/chr.%s.deletions.json' % (deletion_dir, chrom), 'w+') as f:
	json.dump(deletions, f)



