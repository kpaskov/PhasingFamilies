import sys
import numpy as np
import scipy.stats.stats as stats
from itertools import product
import cvxpy as cp
from collections import Counter
import json

# Pull genotypes for a chromosome
data_dir = sys.argv[1] #'../split_gen_ihart_23andme'
ped_file = sys.argv[2] #'../data/160826.ped'
out_file = sys.argv[3]

# ------------------------------------ Basic Info ------------------------------------
chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']

chrom_lengths = {
	'1': 225934550,
	'2': 238204522,
	'3': 194797140,
	'4': 188042934,
	'5': 177695260,
	'6': 167395067,
	'7': 155536559,
	'8': 142964911,
	'9': 120626573,
	'10': 131314747,
	'11': 131169619,
	'12': 130481395,
	'13': 95589878,
	'14': 88289540,
	'15': 81694769,
	'16': 78884753,
	'17': 78129607,
	'18': 74661510,
	'19': 56060841,
	'20': 59505520,
	'21': 35134224,
	'22': 34894566,
	'X': 151100560,
	'Y': 25653566
}

# 0 = 0/0
# 1 = 0/1
# 2 = 1/1
# 3 = ./.
# 4 = -/0 (hemizygous ref)
# 5 = -/1 (hemizygous alt)
# 6 = -/- (double deletion)

errors = [(0, 1), (0, 2), (0, 3), 
          (1, 0), (1, 2), (1, 3), 
          (2, 0), (2, 1), (2, 3), 
          (4, 1), (4, 2), (4, 3),
          (5, 0), (5, 1), (5, 3),
          (6, 0), (6, 1), (6, 2)]
error_to_index = dict([(x, i) for i, x in enumerate(errors)])
print('errors', len(errors))

mendelian_trios = {
    (0, 0, 0), 
    (0, 1, 0), (0, 1, 1),
    (0, 2, 1),
    (1, 0, 0), (1, 0, 1),
    (1, 1, 0), (1, 1, 1), (1, 1, 2),
    (1, 2, 1), (1, 2, 2),
    (2, 0, 1),
    (2, 1, 1), (2, 1, 2),
    (2, 2, 2)
}

mendelian_check = lambda x: x in mendelian_trios
autosome_gen_to_error = dict([(e, e) for e in errors if e[0] <= 2])

print(mendelian_check((0, 0, 0)), mendelian_check((0, 0, 1)))

# ------------------------------------ Pull Data ------------------------------------

# pull sex from ped file
sample_id_to_sex = dict()
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        fam_id, child_id, f_id, m_id = pieces[0:4]

        sample_id_to_sex[m_id] = '2'
        sample_id_to_sex[f_id] = '1'
        if len(pieces) > 4:
            sample_id_to_sex[child_id] = pieces[4]
            
print('sex for inds: %d' % len(sample_id_to_sex))

family_chrom_to_counts = dict()
family_to_inds = dict()
for i, chrom in enumerate(chroms):
    print(chrom, end=' ')
    chrom_length = chrom_lengths[chrom]
    with open('%s/chr.%s.famgen.counts.txt' % (data_dir, chrom), 'r') as f:
        for line in f:
            pieces = line.strip().split('\t')
            famkey, inds = pieces[:2]
            inds = inds.split('.')
            m = len(inds)
            
            if famkey not in family_to_inds:
                family_to_inds[famkey] = inds
            else:
                assert family_to_inds[famkey] == inds
            
            counts = np.zeros((4,)*m, dtype=int)
            for g, c in zip(product([0, 1, 2, 3], repeat=m), pieces[2:]):
                counts[g] = int(c)
                
            if chrom != 'Y':
                counts[(0,)*m] += (chrom_length - np.sum(counts)) 
            else:
                counts[tuple(0 if sample_id_to_sex[ind]=='1' else 3 if sample_id_to_sex[ind]=='2' else None for ind in inds)] += (chrom_length - np.sum(counts)) 
                
            family_chrom_to_counts[(famkey, chrom)] = counts

print('Families of each size', Counter([len(inds) for fkey, inds in family_to_inds.items()]))
famkeys = sorted(set([x[0] for x in family_chrom_to_counts.keys()]))
print('Families', len(famkeys))


# ------------------------------------ Poisson Regression ------------------------------------

def get_mendelian(ind_is_mendelian):
    
    # differentiate mendelian and non-mendelian famgens
    m = len(ind_is_mendelian)
    is_mendelian = np.ones((4,)*m, dtype=bool)
    for famgen in product([0, 1, 2, 3], repeat=m):
        is_mend = True
        for j in range(2, m):
            if not ind_is_mendelian[j](tuple(famgen[x] for x in [0, 1, j])):
                is_mend = False
        is_mendelian[famgen] = is_mend
    return is_mendelian

def get_error_to_famgen_pairs(is_mendelian, ind_gen_switch_to_error):
    nonmendelian_famgens = list(zip(*np.where(~is_mendelian)))
    #print('Mendelian', np.sum(is_mendelian), 'Nonmendelian', len(nonmendelian_famgens))

    error_to_fg_pairs = dict([(e, []) for e in errors])
    for nmfg in nonmendelian_famgens:
        for j in range(m):
            for i in range(4):
                mfg = tuple(i if k==j else nmfg[k] for k in range(m))
                if is_mendelian[mfg]:
                    error = ind_gen_switch_to_error[j][(mfg[j], nmfg[j])]
                    error_to_fg_pairs[error].append((nmfg, mfg))
        
    return nonmendelian_famgens, error_to_fg_pairs

def build_family_chrom_X_and_y(counts, nonmendelian_famgens, error_to_fg_pairs):
    m = len(counts.shape)
    nm_famgen_to_index = dict([(x, i) for i, x in enumerate(nonmendelian_famgens)])
    
    X = np.zeros((len(nonmendelian_famgens), len(errors)))
    
    for error, fg_pairs in error_to_fg_pairs.items():
        error_index = error_to_index[error]
        for famgen, neighbor in fg_pairs: # nmfg, mfg
            famgen_index = nm_famgen_to_index[famgen]
            
            if counts[neighbor]>0:
                X[famgen_index, error_index] += counts[neighbor]
                
    y = np.asarray([counts[x] for x in nonmendelian_famgens])
    
    return X, y

chrom_Xs, chrom_ys = [[] for chrom in chroms], [[] for chrom in chroms]

for famkey in famkeys:
    inds = family_to_inds[famkey]
    m = len(inds)
    
    is_mendelian = get_mendelian([None, None] + ([mendelian_check]*(m-2)))

    nonmendelian_famgens, error_to_fg_pairs = get_error_to_famgen_pairs(is_mendelian, [autosome_gen_to_error]*m)
    
    for i, chrom in enumerate(chroms):
        if chrom != 'X' and chrom != 'Y':
            counts = family_chrom_to_counts[(famkey, chrom)]
            X, y = build_family_chrom_X_and_y(counts, nonmendelian_famgens, error_to_fg_pairs)
    
            chrom_Xs[i].append(X)
            chrom_ys[i].append(y)
            
        elif chrom == 'X':
            # X has its own rules for mendelian/non-mendelian
            ind_is_mendelian = [None, None] + [mendelian_X_F_check if sample_id_to_sex[ind] == '2' else mendelian_X_M_check if sample_id_to_sex[ind] == '1' else None for ind in inds[2:]]
            X_is_mendelian = get_mendelian(ind_is_mendelian)
            ind_gen_switch_to_error = [X_gen_to_error_F if sample_id_to_sex[ind] == '2' else X_gen_to_error_M if sample_id_to_sex[ind] == '1' else None for ind in inds]
            X_nonmendelian_famgens, X_error_to_fg_pairs = get_error_to_famgen_pairs(X_is_mendelian, ind_gen_switch_to_error)

            counts = family_chrom_to_counts[(famkey, chrom)]
            X, y = build_family_chrom_X_and_y(counts, X_nonmendelian_famgens, X_error_to_fg_pairs)

            chrom_Xs[i].append(X)
            chrom_ys[i].append(y)
            
        elif chrom == 'Y':
            # Y has its own rules for mendelian/non-mendelian
            ind_is_mendelian = [None, None] + [mendelian_Y_F_check if sample_id_to_sex[ind] == '2' else mendelian_Y_M_check if sample_id_to_sex[ind] == '1' else None for ind in inds[2:]]
            Y_is_mendelian = get_mendelian(ind_is_mendelian)
            Y_nonmendelian_famgens, Y_error_to_fg_pairs = get_error_to_famgen_pairs(Y_is_mendelian, [Y_gen_to_error]*m)
            
            counts = family_chrom_to_counts[(famkey, chrom)]
            X, y = build_family_chrom_X_and_y(counts, Y_nonmendelian_famgens, Y_error_to_fg_pairs)
            
            chrom_Xs[i].append(X)
            chrom_ys[i].append(y)
            
            
def estimate_family_error(X, y, init=None):
    norm = np.max(X)
    X_norm = X/norm
    
    # cvxpy
    n = cp.Variable(X_norm.shape[1])
    if init is not None:
        n.value = init

    mu = np.sum(X_norm, axis=0)
    objective = cp.Minimize(mu*n - y*cp.log(X_norm*n))
    
    _, upper = scipy.stats.poisson.interval(0.95, y[y>0])
    constraints = [n>=0, X_norm[y>0, :]*n <= upper]
    prob = cp.Problem(objective, constraints)
    
    result = prob.solve(solver='ECOS', max_iters=1000)
    print(prob.status)
    
    n = np.asarray([v[0, 0] for v in n.value])
    
    return prob.status, n/norm, X_norm.dot(n), y

# sum by family
famsum_genome_X, famsum_genome_y = [], []
for i in range(len(famkeys)):
    famsum_genome_X.append(np.sum(np.asarray([chrom_Xs[j][i] for j in range(22)]), axis=0))
    famsum_genome_y.append(np.sum(np.asarray([chrom_ys[j][i] for j in range(22)]), axis=0))
    
    # append Xchrom
    famsum_genome_X.append(chrom_Xs[22][i])
    famsum_genome_y.append(chrom_ys[22][i])
    
    # append Ychrom
    famsum_genome_X.append(chrom_Xs[23][i])
    famsum_genome_y.append(chrom_ys[23][i])
    
famsum_genome_X, famsum_genome_y = np.vstack(famsum_genome_X), np.hstack(famsum_genome_y)

print('Removing zero cols:', [errors[i] for i in np.where(np.sum(famsum_genome_X, axis=0)==0)[0]])
famsum_genome_X = famsum_genome_X[:, np.sum(famsum_genome_X, axis=0)>0]

print('Removing zero rows:', np.sum(np.sum(famsum_genome_X, axis=1)==0))
indices = np.where(np.sum(famsum_genome_X, axis=1) != 0)[0]
famsum_genome_X = famsum_genome_X[indices, :]
famsum_genome_y = famsum_genome_y[indices]
print(famsum_genome_X.shape, famsum_genome_y.shape)
    
prob_status, famsum_genome_n, famsum_genome_exp, famsum_genome_obs = estimate_family_error(famsum_genome_X, famsum_genome_y)

baseline = np.ones((7,))
for e, c in zip(errors, famsum_genome_n):
    #print(e, -np.log10(c))
    baseline[e[0]] -= c

# estimate probability of recombination
mat_crossover = -(np.log10(22.8)-np.log10(sum(chrom_lengths.values())))
pat_crossover = -(np.log10(1.7*22.8)-np.log10(sum(chrom_lengths.values())))

num_deletions = 1
del_trans = -(np.log10(2*num_deletions)-np.log10(sum(chrom_lengths.values())))

num_hts = 1000
hts_trans = -(np.log10(2*num_hts)-np.log10(sum(chrom_lengths.values())))

params = {
	"-log10(P[deletion_entry_exit])": del_trans,
	"-log10(P[maternal_crossover])": mat_crossover,
	"-log10(P[paternal_crossover])": pat_crossover,
	"-log10(P[hard_to_seq_region_entry_exit])": hts_trans,

	"-log10(P[obs=./.|true_gen=0/0])": -np.log10(famsum_genome_n[error_to_index[(0, 3)]]),
	"-log10(P[obs=0/0|true_gen=0/0])": -np.log10(baseline[0]),
	"-log10(P[obs=0/1|true_gen=0/0])": -np.log10(famsum_genome_n[error_to_index[(0, 1)]]),
	"-log10(P[obs=1/1|true_gen=0/0])": -np.log10(famsum_genome_n[error_to_index[(0, 2)]]),

	"-log10(P[obs=./.|true_gen=0/1])": -np.log10(famsum_genome_n[error_to_index[(1, 3)]]),
	"-log10(P[obs=0/0|true_gen=0/1])": -np.log10(famsum_genome_n[error_to_index[(1, 0)]]),
	"-log10(P[obs=0/1|true_gen=0/1])": -np.log10(baseline[1]),
	"-log10(P[obs=1/1|true_gen=0/1])": -np.log10(famsum_genome_n[error_to_index[(1, 2)]]),

	"-log10(P[obs=./.|true_gen=1/1])": -np.log10(famsum_genome_n[error_to_index[(2, 3)]]),
	"-log10(P[obs=0/0|true_gen=1/1])": -np.log10(famsum_genome_n[error_to_index[(2, 0)]]),
	"-log10(P[obs=0/1|true_gen=1/1])": -np.log10(famsum_genome_n[error_to_index[(2, 1)]]),
	"-log10(P[obs=1/1|true_gen=1/1])": -np.log10(baseline[2]),

	"-log10(P[obs=./.|true_gen=-/0])": -np.log10(famsum_genome_n[error_to_index[(4, 3)]]),
	"-log10(P[obs=0/0|true_gen=-/0])": -np.log10(baseline[4]),
	"-log10(P[obs=0/1|true_gen=-/0])": -np.log10(famsum_genome_n[error_to_index[(4, 1)]]),
	"-log10(P[obs=1/1|true_gen=-/0])": -np.log10(famsum_genome_n[error_to_index[(4, 2)]]),

	"-log10(P[obs=./.|true_gen=-/1])": -np.log10(famsum_genome_n[error_to_index[(5, 3)]]),
	"-log10(P[obs=0/0|true_gen=-/1])": -np.log10(famsum_genome_n[error_to_index[(5, 0)]]),
	"-log10(P[obs=0/1|true_gen=-/1])": -np.log10(famsum_genome_n[error_to_index[(5, 1)]]),
	"-log10(P[obs=1/1|true_gen=-/1])": -np.log10(baseline[5]),

	"-log10(P[obs=./.|true_gen=-/-])": -np.log10(baseline[6]),
	"-log10(P[obs=0/0|true_gen=-/-])": -np.log10(famsum_genome_n[error_to_index[(6, 0)]]),
	"-log10(P[obs=0/1|true_gen=-/-])": -np.log10(famsum_genome_n[error_to_index[(6, 1)]]),
	"-log10(P[obs=1/1|true_gen=-/-])": -np.log10(famsum_genome_n[error_to_index[(6, 2)]]),

	"x-times higher probability of error in hard-to-sequence region": 10
}

with open(out_file, 'w+') as f:
	json.dump(params, f)

