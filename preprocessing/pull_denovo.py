from collections import namedtuple, defaultdict, Counter
import json
import numpy as np
import matplotlib.pyplot as plt
import gzip
from os import listdir
from scipy import sparse

data_dir = '../DATA/ssc.hg38'
ped_file = '../DATA/ssc.hg38/ssc.ped'
chroms = ['10']
#chroms = [str(x) for x in range(1, 23)]

sample_file = '%s/genotypes/samples.json' % data_dir
# pull samples
with open(sample_file, 'r') as f:
    individuals = json.load(f)
sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(individuals)])

child_to_mother = dict()
child_to_father = dict()
child_to_phen = dict()
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) < 6:
            print('ped parsing error', line)
        else:
            fam_id, child_id, f_id, m_id, _, phen = pieces[0:6]
                
            if f_id != '0' and f_id in sample_id_to_index:
                child_to_father[child_id] = f_id
            if m_id != '0' and m_id in sample_id_to_index:
                child_to_mother[child_id] = m_id
                    
            child_to_phen[child_id] = phen
            
children = sorted(set(child_to_mother.keys()) & set(child_to_father.keys() & set(sample_id_to_index.keys())))
print('children', len(children))

child_indices = [sample_id_to_index[x] for x in children]
father_indices = np.array([sample_id_to_index[child_to_father[x]] for x in children], dtype=int)
mother_indices = np.array([sample_id_to_index[child_to_mother[x]] for x in children], dtype=int)

with open('possible_denovo_ssc.txt', 'w+') as f:
    f.write('\t'.join(['chrom', 'pos', 'child', 'AF', 'VCF']) + '\n')
    for chrom in chroms:
        gen_files = sorted([f for f in listdir('%s/genotypes' % data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f], key=lambda x: int(x.split('.')[2]))
        coord_files = sorted([f for f in listdir('%s/genotypes' % data_dir) if ('chr.%s.' % chrom) in f and 'gen.coordinates.npy' in f], key=lambda x: int(x.split('.')[2]))
        annot_files = sorted([f for f in listdir('%s/genotypes' % data_dir) if ('chr.%s.' % chrom) in f and 'gen.variants.txt.gz' in f], key=lambda x: int(x.split('.')[2]))
        assert len(gen_files) == len(coord_files)
        assert len(gen_files) == len(annot_files)

        for gen_file, coord_file, annot_file in zip(gen_files, coord_files, annot_files):
            print(gen_file)
            with gzip.open('%s/genotypes/%s' % (data_dir, annot_file), 'rt') as annot_f:
                annotations = [line for line in annot_f]

            af = [dict([tuple(y.split('=')) for y in x.strip().split()[7].split(';') if '=' in y]).get('AF', np.nan) for x in annotations]
            af = np.array([float(x) if ',' not in x else np.nan for x in af])

            coords = np.load('%s/genotypes/%s' % (data_dir, coord_file))
            gen = sparse.load_npz('%s/genotypes/%s' % (data_dir, gen_file))

            poss = coords[:, 1]
            is_snp = coords[:, 2]==1

            #remove indels
            is_ok = is_snp

            # remove multiallelic sites
            multi_indices = np.where(coords[1:, 1]==coords[:-1, 1])[0]
            is_ok[multi_indices] = False
            is_ok[multi_indices+1] = False

            # remove AF > 0.01
            is_ok[af>0.01] = False        

            possible_denovo_children, possible_denovo_positions = (gen[child_indices, :]==1).nonzero()
            
            indices = ((gen[mother_indices[possible_denovo_children], possible_denovo_positions]==0) & \
                      (gen[father_indices[possible_denovo_children], possible_denovo_positions]==0) & \
                      is_ok[possible_denovo_positions]).A.flatten()
            print('de novos', np.sum(indices))

            for child_index, index in zip(possible_denovo_children[indices], possible_denovo_positions[indices]):
                f.write('%s\t%s' % (children[child_index], annotations[index]))

