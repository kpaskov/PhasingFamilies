import sys
from itertools import product
from os import listdir
import numpy as np
import scipy.sparse as sparse


data_dir = sys.argv[1]
ped_file = sys.argv[2]
chrom = sys.argv[3]

if chrom == '23':
    chrom = 'X'
if chrom == '24':
    chrom = 'Y'
if chrom == '25':
    chrom = 'MT'

sample_file = '%s/chr.%s.gen.samples.txt' % (data_dir, chrom)
out_file = '%s/chr.%s.famgen.counts.txt' % (data_dir, chrom)
regions, include = None, None
if len(sys.argv) > 4:
    if sys.argv[4] == '--include':
        include = True
    elif sys.argv[4] == '--exclude':
        include = False
    else:
        raise Exception('Bad arguments.')
    bed_file = sys.argv[5] 

    regions = []
    with open(bed_file, 'r') as f:
        for line in f:
            pieces = line.strip().split('\t')
            if pieces[0] == chrom:
                regions.append(int(pieces[1]))
                regions.append(int(pieces[2])+1)
    regions = np.array(regions)

    if include:
        print('including %d regions' % int(len(regions)/2))
    else:
        print('excluding %d regions' % int(len(regions)/2))

    out_dir = sys.argv[6]
    out_file = '%s/chr.%s.famgen.counts.txt' % (out_dir, chrom)
    print('saving to %s' % out_file)


# pull families with sequence data
with open(sample_file, 'r') as f:
    sample_id_to_index = dict([(line.strip(), i) for i, line in enumerate(f)])
with open(sample_file, 'r') as f:
    sample_ids = [line.strip() for line in f]

# pull families from ped file
families = dict()

with open(ped_file, 'r') as f:	
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) < 4:
            print('ped parsing error', line)
        else:
            fam_id, child_id, f_id, m_id = pieces[0:4]

            if child_id in sample_id_to_index and f_id in sample_id_to_index and m_id in sample_id_to_index:
                if (fam_id, m_id, f_id) not in families:
                    families[(fam_id, m_id, f_id)] = [m_id, f_id]
                families[(fam_id, m_id, f_id)].append(child_id)
print('families %d' % len(families))

with open(out_file, 'w+') as f:	
    gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f])

    # pull snp positions
    pos_data = np.load('%s/chr.%s.gen.coordinates.npy' % (data_dir, chrom))
    is_snp = pos_data[:, 2].astype(bool)
    is_pass = pos_data[:, 3].astype(bool)

    if regions is not None:
        insert_loc = np.searchsorted(regions, pos_data[:, 1])
        if include:
            is_ok_region = np.remainder(insert_loc, 2)==1
        else:
            is_ok_region = np.remainder(insert_loc, 2)==0
    else:
        is_ok_region = np.ones(is_snp.shape, dtype=bool)


    print(np.sum(~is_snp))
    print(np.sum(~is_pass))
    print(np.sum(~is_ok_region))

    # Pull data together
    A = sparse.hstack([sparse.load_npz('%s/%s' % (data_dir, gen_file)) for gen_file in gen_files])

    # filter out snps
    A = A[:, is_snp & is_pass & is_ok_region]
    print('genotype matrix prepared', A.shape)

    for famkey, inds in families.items():
        m = len(inds)
        genotype_to_counts = np.zeros((4,)*m, dtype=int)
        indices = [sample_id_to_index[ind] for ind in inds]
        family_genotypes = A[indices, :].A
        
        # remove positions where whole family is homref
        all_hom_ref = np.all(family_genotypes==0, axis=0)
        family_genotypes = family_genotypes[:, ~all_hom_ref]
        #print(famkey, family_genotypes.shape)

        # recode missing values
        family_genotypes[family_genotypes<0] = 3
        
        # fill in genotype_to_counts
        unique_gens, counts = np.unique(family_genotypes, return_counts=True, axis=1)
        for g, c in zip(unique_gens.T, counts):
            genotype_to_counts[tuple(g)] += c
        genotype_to_counts[(0,)*m] = np.sum(all_hom_ref)

        # write to file
        f.write('\t'.join(['.'.join(famkey), '.'.join(inds)] + \
            [str(genotype_to_counts[g]) for g in product([0, 1, 2, 3], repeat=m)]) + '\n')
        
