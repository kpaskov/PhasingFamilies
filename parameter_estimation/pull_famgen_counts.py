import sys
from itertools import product
from os import listdir
import numpy as np
import scipy.sparse as sparse


data_dir = sys.argv[1]
ped_file = sys.argv[2]
chrom = sys.argv[3]

sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
out_file = '%s/chr.%s.famgen.counts.txt' % (data_dir, chrom)

# pull families with sequence data
with open(sample_file, 'r') as f:
    sample_id_to_index = dict([(line.strip(), i) for i, line in enumerate(f)])
with open(sample_file, 'r') as f:
    sample_ids = [line.strip() for line in f]

# pull families from ped file
families = dict()

with open(ped_file, 'r') as f:	
    for line in f:
        pieces = line.strip().split()
        fam_id, child_id, f_id, m_id = pieces[0:4]

        if child_id in sample_id_to_index and f_id in sample_id_to_index and m_id in sample_id_to_index and 'LCL' not in child_id:
            if (fam_id, m_id, f_id) not in families:
                families[(fam_id, m_id, f_id)] = [m_id, f_id]
            families[(fam_id, m_id, f_id)].append(child_id)
print('families %d' % len(families))

with open(out_file, 'w+') as f:	
    gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])

    # pull snp positions
    pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))
    is_snp = pos_data[:, 2].astype(bool)

    # Pull data together
    A = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

    # filter out snps
    A = A[:, is_snp]
    print('genotype matrix prepared', A.shape)

    for famkey, inds in families.items():
        m = len(inds)
        genotype_to_counts = np.zeros((4,)*m, dtype=int)
        indices = [sample_id_to_index[ind] for ind in inds]
        family_genotypes = A[indices, :].A
        
        # remove positions where whole family is homref
        ok_indices = np.any(family_genotypes!=0, axis=0)
        family_genotypes = family_genotypes[:, ok_indices]
        #print(famkey, family_genotypes.shape)

        # recode missing values
        family_genotypes[family_genotypes<0] = 3
        
        # fill in genotype_to_counts
        unique_gens, counts = np.unique(family_genotypes, return_counts=True, axis=1)
        for g, c in zip(unique_gens.T, counts):
            genotype_to_counts[tuple(g)] += c
        genotype_to_counts[(0,)*m] = A.shape[1]-np.sum(genotype_to_counts)

        # write to file
        f.write('\t'.join(['.'.join(famkey), '.'.join(inds)] + \
            [str(genotype_to_counts[g]) for g in product([0, 1, 2, 3], repeat=m)]) + '\n')
        
