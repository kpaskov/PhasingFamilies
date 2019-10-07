import sys
import gzip
import numpy as np
import scipy.sparse as sparse
from os import listdir

data_dir = sys.argv[1]

pass_from_gen = False
if len(sys.argv)>2 and sys.argv[2] == '--pass_from_gen':
    pass_from_gen = True
    ped_file = sys.argv[3]
    print('Generate PASS from genotypes.')

chroms = [str(x) for x in range(1, 23)] + ['X', 'Y']

def calculate_af_and_percent_miss(chrom, indices=None):
    # load genotypes
    gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])

    # Pull data together
    A = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

    if indices is None:
        indices = np.ones((A.shape[0]), dtype=bool)

    # AF filter / missing filter
    homalt = (A[indices, :]==2).sum(axis=0).A.flatten()
    het = (A[indices, :]==1).sum(axis=0).A.flatten()
    miss = (A[indices, :]<0).sum(axis=0).A.flatten()

    af = np.ones(homalt.shape)
    af[np.sum(indices) > miss] = (2*homalt[np.sum(indices) > miss] + het[np.sum(indices) > miss])/(2*np.sum(indices) - 2*miss[np.sum(indices) > miss])
    percent_miss = miss/np.sum(indices)

    return af, percent_miss


for chrom in chroms:
    print(chrom, end=' ')

    # pull snp positions
    pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))  
    
    if pass_from_gen:
        # pull male/female indices
        with open('%s/chr.%s.gen.samples.txt' % (data_dir, chrom), 'r') as f:
            sample_ids = [x.strip() for x in f]

        sample_id_to_sex = dict()
        # Sex (1=male; 2=female; other=unknown)
        with open(ped_file, 'r') as f:
            for line in f:
                pieces = line.strip().split('\t')
                if len(pieces) >= 6:
                    fam_id, sample_id, f_id, m_id, sex, disease_status = pieces[0:6]
                    sample_id_to_sex[sample_id] = sex
        is_male = np.array([x in sample_id_to_sex and sample_id_to_sex[x]=='1' for x in sample_ids], dtype=bool)
        is_female = np.array([x in sample_id_to_sex and sample_id_to_sex[x]=='2' for x in sample_ids], dtype=bool)

        if chrom == 'X':
            af_f, percent_miss_f = calculate_af_and_percent_miss(chrom, indices=is_female)
            af_m, percent_miss_m = calculate_af_and_percent_miss(chrom, indices=is_male)

            is_pass = (percent_miss_f < 0.1) & (percent_miss_m < 0.2)
            
        elif chrom == 'Y':
            af_m, percent_miss_m = calculate_af_and_percent_miss(chrom, indices=is_male)
            is_pass = (percent_miss_m < 0.2)
        else:
            af, percent_miss = calculate_af_and_percent_miss(chrom)
            is_pass = (percent_miss < 0.1)

    else:
        is_pass = []
        with gzip.open('%s/chr.%s.gen.variants.txt.gz' % (data_dir, chrom), 'rt') as f:
            for line in f:
                pieces = line.strip().split('\t')
                is_pass.append(pieces[6] == 'PASS')
        is_pass = np.array(is_pass)
            
      

    chrom_int = 23 if chrom == 'X' else 24 if chrom == 'Y' else int(chrom)
    np.save('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom), np.hstack((chrom_int*np.ones((pos_data.shape[0], 1), dtype=int), pos_data[:, 1:3], is_pass[:, np.newaxis])))

