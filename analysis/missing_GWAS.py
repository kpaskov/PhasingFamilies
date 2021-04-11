import json
import numpy as np
from collections import defaultdict
import random
from os import listdir
from scipy import sparse
import scipy.stats as stats
import statsmodels.api as sm

import argparse

parser = argparse.ArgumentParser(description='Run missing data GWAS.')
parser.add_argument('ped_file', type=str, help='Ped file of family structure.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data in .npy format.')
parser.add_argument('chrom', type=str, help='Chromosome.')
parser.add_argument('out_dir', type=str, help='Output directory.')

args = parser.parse_args()

with open('%s/samples.json' % args.data_dir, 'r') as f:
    sample_ids = json.load(f)
print(len(sample_ids))
sample_id_to_index = dict([(x, i) for i, x in enumerate(sample_ids)])

# pull affected status
# (0=unknown; 1=unaffected; 2=affected)

child_id_to_sex = dict()
child_id_to_affected = dict()
family_to_people = defaultdict(list)

def load_ped(ped_file):
    with open(ped_file, 'r') as f:
        for line in f:
            pieces = line.strip().split('\t')
            if len(pieces) >= 6:
                fam_id, child_id, f_id, m_id, sex, disease_status = pieces[0:6]
                child_id_to_affected[child_id] = disease_status
                child_id_to_sex[child_id] = sex
                family_to_people[(fam_id, m_id, f_id)].append(child_id)
load_ped(args.ped_file)

children = []
for (fam_id, m_id, f_id), cs in family_to_people.items():
    cs = [x for x in cs if x in sample_id_to_index]
    children.extend(cs)

child_is_male = [1 if child_id_to_sex[child]=='1' else 0 for child in children]
child_is_affected = [1 if child_id_to_affected[child]=='2' else 0 for child in children]
print(len(children))

child_indices = [sample_id_to_index[x] for x in children]

child_is_affected = np.array(child_is_affected, dtype=bool)
child_is_male = np.array(child_is_male, dtype=bool)
num_affected = np.sum(child_is_affected)
num_unaffected = np.sum(~child_is_affected)

gens_child = []
pos_coord = []

gen_files = sorted([f for f in listdir(args.data_dir) if ('chr.%s.' % args.chrom) in f and 'gen.npz' in f], key=lambda x: int(x.split('.')[2]))
coord_files = sorted([f for f in listdir(args.data_dir) if ('chr.%s.' % args.chrom) in f and 'gen.coordinates.npy' in f], key=lambda x: int(x.split('.')[2]))
af_files = sorted([f for f in listdir(args.data_dir) if ('chr.%s.' % args.chrom) in f and 'gen.af.npy' in f], key=lambda x: int(x.split('.')[2]))
    
for gen_file, coord_file, af_file in zip(gen_files, coord_files, af_files):
    coords = np.load('%s/%s' % (args.data_dir, coord_file))

    if coords.shape[0]>0:
        poss = coords[:, 1]
        is_snp = coords[:, 2]==1
        is_pass = coords[:, 3]==1
            
        gen = sparse.load_npz('%s/%s' % (args.data_dir, gen_file))[child_indices, :]
        has_missing = ((gen<0).sum(axis=0)>10).A.flatten()

        indices = is_snp & has_missing
        print(np.sum(indices), end=' ')
        if np.sum(indices)>0:
            gens_child.append((gen[:, indices]<0).A)
            pos_coord.append(poss[indices])

np.hstack(pos_coord).dump('%s/chr.%s.coords.npy' % (args.out_dir, args.chrom))

num_missing = np.zeros((len(children),))
num_missing_aff = []
num_missing_unaff = []
for g in gens_child:
    num_missing += np.sum(g, axis=1)
    num_missing_aff.append(np.sum(g[child_is_affected, :], axis=0)/np.sum(child_is_affected))
    num_missing_unaff.append(np.sum(g[~child_is_affected, :], axis=0)/np.sum(~child_is_affected))

np.hstack(num_missing_aff).dump('%s/chr.%s.missing_aff.npy' % (args.out_dir, args.chrom))
np.hstack(num_missing_unaff).dump('%s/chr.%s.missing_unaff.npy' % (args.out_dir, args.chrom))

def lr_pvalue(missing):
    if np.sum(missing)>10 and np.sum(~missing)>10:
        try:
            # fill in the first column of mds_data with the genomic data
            X = np.zeros((missing.shape[0], 4))
            y = np.zeros((missing.shape[0],))

            X[:, 0] = missing
            X[:, 1] = num_missing
            X[:, 2] = child_is_male
            X[:, 3] = 1

            y = child_is_affected

            model = sm.Logit(y, X)
            result = model.fit(disp=False)
            if result.mle_retvals["converged"]:
                return result.pvalues[0], result.params[0]
            
        except np.linalg.LinAlgError:
            return np.nan, np.nan
    return np.nan, np.nan

pvalues = []
coeffs = []
for j, g in enumerate(gens_child):
    print(j, end=' ')
    ps = np.ones((g.shape[1],))
    cs = np.ones((g.shape[1],))
    missing_aff = np.sum(g[child_is_affected, :], axis=0)
    missing_unaff = np.sum(g[~child_is_affected, :], axis=0)
    print(g.shape[1], end=' ')
    for i in np.arange(g.shape[1]):
        ps[i], cs[i] = lr_pvalue(g[:, i])
        #cs[i], ps[i] = stats.chi2_contingency([[missing_aff[i], num_affected-missing_aff[i]],
        #                                [missing_unaff[i], num_unaffected-missing_unaff[i]]])[:2]
        if i%1000==0:
            print(i, end=' ')
    pvalues.append(ps)
    coeffs.append(cs)

np.hstack(pvalues).dump('%s/chr.%s.pvalues.npy' % (args.out_dir, args.chrom))
np.hstack(coeffs).dump('%s/chr.%s.coeffs.npy' % (args.out_dir, args.chrom))

