import sys
import numpy as np
import scipy.stats
from itertools import product
import cvxpy as cp
from collections import Counter
import json

# Pull genotypes for a chromosome
data_dir = sys.argv[1] #'../split_gen_ihart_23andme'
ped_file = sys.argv[2] #'../data/160826.ped'
out_file = sys.argv[3] # ../parameter_estimation/23andme_params.json


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
print('num error types', len(errors))

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

mendelian_trios_X_F = {
    (0, 0, 0), 
    (0, 2, 1),
    (1, 0, 0), (1, 0, 1),
    (1, 2, 1), (1, 2, 2),
    (2, 0, 1),
    (2, 2, 2)
}
mendelian_trios_X_M = {
    (0, 0, 0), 
    (0, 2, 0),
    (1, 0, 0), (1, 0, 2),
    (1, 2, 0), (1, 2, 2),
    (2, 0, 2),
    (2, 2, 2)
}

mendelian_X_F_check = lambda x: x in mendelian_trios_X_F
mendelian_X_M_check = lambda x: x in mendelian_trios_X_M

X_gen_to_error_F = autosome_gen_to_error
X_gen_to_error_M = {(0, 1): (4, 1), (0, 2): (4, 2), (0, 3): (4, 3),
                    (2, 0): (5, 0), (2, 1): (5, 1), (2, 3): (5, 3)}

mendelian_trios_Y_F = {
    (3, 0, 3), 
    (3, 2, 3)
}
mendelian_trios_Y_M = {
    (3, 0, 0), 
    (3, 2, 2),
}
mendelian_Y_F_check = lambda x: x in mendelian_trios_Y_F
mendelian_Y_M_check = lambda x: x in mendelian_trios_Y_M

Y_gen_to_error = {(3, 0): (6, 0), (3, 1): (6, 1), (3, 2): (6, 2),
                  (0, 1): (4, 1), (0, 2): (4, 2), (0, 3): (4, 3),
                  (2, 0): (5, 0), (2, 1): (5, 1), (2, 3): (5, 3)}


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
            
print('pulled sex for %d inds' % len(sample_id_to_sex))

family_chrom_to_counts = dict()
family_to_inds = dict()
for i, chrom in enumerate(chroms):
    print(chrom, end=' ')
    
    with open('%s/chr.%s.famgen.counts.txt' % (data_dir, chrom), 'r') as f:
        for line in f:
            pieces = line.strip().split('\t')
            famkey, inds = pieces[:2]
            
            if 'ssc' in data_dir:
            	# unfortunately, ssc uses . in their sample names
                inds = inds.split('.')
                inds = ['%s.%s' % (inds[i], inds[i+1]) for i in range(0, len(inds), 2)]
            else:
                inds = inds.split('.')

            m = len(inds)

            if m<=7:
                if famkey not in family_to_inds:
                    family_to_inds[famkey] = inds
                else:
                    assert family_to_inds[famkey] == inds
                
                counts = np.zeros((4,)*m, dtype=int)
                for g, c in zip(product([0, 1, 2, 3], repeat=m), pieces[2:]):
                    counts[g] = int(c)
                    
                family_chrom_to_counts[(famkey, chrom)] = counts

print('Families of each size', Counter([len(inds) for fkey, inds in family_to_inds.items()]))
famkeys = sorted(set([x[0] for x in family_chrom_to_counts.keys()]))
famkeys = [k for k in famkeys if np.all([ind in sample_id_to_sex for ind in family_to_inds[k]])]
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
            if None in ind_is_mendelian[2:]:
            	print(inds, [sample_id_to_sex[x] for x in inds])
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
    print('Estimating...', X.shape, y.shape)
    alpha = 1.0/np.max(X)
    
    # cvxpy
    n = cp.Variable(X.shape[1])
    if init is not None:
        n.value = init

    mu = np.sum(alpha*X, axis=0)
    objective = cp.Minimize(mu*n - alpha*y*cp.log(alpha*X*n))
    
    #upper = 0.5*scipy.stats.chi2.ppf(0.95, (2*y) + 2)
    constraints = [n>=0]#, X[y>0, :]*n <= upper[y>0]]
    prob = cp.Problem(objective, constraints)
    
    result = prob.solve(solver='ECOS', max_iters=1000)
    print(prob.status)
    
    print(n.value, n.value.shape)
    n = np.asarray([v for v in n.value])
    
    return prob.status, n, X.dot(n), y

# ------------------------------------ Estimate Error Rates ------------------------------------

# detect outliers
sd_cutoff = scipy.stats.norm.isf(0.05/len(famkeys))
outliers = set()
for j in range(len(chroms)):
    family_errors = np.array([np.sum(chrom_ys[j][i]) for i in range(len(famkeys))])
    median = np.median(family_errors)
    med_abs_deviation = np.median(np.abs(family_errors-median))
    
    outlier_indices = (family_errors > median+(sd_cutoff*med_abs_deviation/0.6745)) | (family_errors < median-(sd_cutoff*med_abs_deviation/0.6745))
    outliers.update(np.where(outlier_indices)[0])
    print('chrom %s outliers %d' % (chroms[j], np.sum(outlier_indices)))
print('Total outliers removed', len(outliers))

# build X and y
famsum_genome_X, famsum_genome_y = [], []
for i in range(len(famkeys)):
    if i not in outliers:
        famsum_genome_X.append(np.sum(np.asarray([chrom_Xs[j][i] for j in range(22)]), axis=0))
        famsum_genome_y.append(np.sum(np.asarray([chrom_ys[j][i] for j in range(22)]), axis=0))

        famsum_genome_X.append(chrom_Xs[22][i])
        famsum_genome_y.append(chrom_ys[22][i])

        famsum_genome_X.append(chrom_Xs[23][i])
        famsum_genome_y.append(chrom_ys[23][i])
       
famsum_genome_X, famsum_genome_y = np.vstack(famsum_genome_X), np.hstack(famsum_genome_y)

print('Removing zero cols:', [errors[i] for i in np.where(np.sum(famsum_genome_X, axis=0)==0)[0]])
famsum_genome_X = famsum_genome_X[:, np.sum(famsum_genome_X, axis=0)>0]

print('Removing zero rows:', np.sum(np.sum(famsum_genome_X, axis=1)==0))
indices = np.where(np.sum(famsum_genome_X, axis=1) != 0)[0]
famsum_genome_X = famsum_genome_X[indices, :]
famsum_genome_y = famsum_genome_y[indices]

#print('Removing rows with y=0:', np.sum(famsum_genome_y==0))
#indices = np.where(famsum_genome_y > 0)[0]
#famsum_genome_X = famsum_genome_X[indices, :]
#famsum_genome_y = famsum_genome_y[indices]

print(famsum_genome_X.shape, famsum_genome_y.shape)
    
prob_status, famsum_genome_n, famsum_genome_exp, famsum_genome_obs = estimate_family_error(famsum_genome_X, famsum_genome_y)

# ------------------------------------ Write to file ------------------------------------

error_estimates = np.maximum(famsum_genome_n.tolist(), 10.0**-10)

baseline = np.ones((7,))
for e, c in zip(errors, error_estimates):
    #print(e, -np.log10(c))
    baseline[e[0]] -= c

# estimate probability of recombination
mat_crossover = -(np.log10(22.8)-np.log10(sum(chrom_lengths.values())))
pat_crossover = -(np.log10(1.7*22.8)-np.log10(sum(chrom_lengths.values())))

num_deletions = 100
del_trans = -(np.log10(2*num_deletions)-np.log10(sum(chrom_lengths.values())))

num_hts = 1000
hts_trans = -(np.log10(2*num_hts)-np.log10(sum(chrom_lengths.values())))

params = {
	"-log10(P[deletion_entry_exit])": del_trans,
	"-log10(P[maternal_crossover])": mat_crossover,
	"-log10(P[paternal_crossover])": pat_crossover,
	"-log10(P[hard_to_seq_region_entry_exit])": hts_trans,
    "-log10(P[low_coverage_region_entry_exit])": hts_trans,

	"-log10(P[obs=./.|true_gen=0/0])": -np.log10(error_estimates[error_to_index[(0, 3)]]),
	"-log10(P[obs=0/0|true_gen=0/0])": -np.log10(baseline[0]),
	"-log10(P[obs=0/1|true_gen=0/0])": -np.log10(error_estimates[error_to_index[(0, 1)]]),
	"-log10(P[obs=1/1|true_gen=0/0])": -np.log10(error_estimates[error_to_index[(0, 2)]]),

	"-log10(P[obs=./.|true_gen=0/1])": -np.log10(error_estimates[error_to_index[(1, 3)]]),
	"-log10(P[obs=0/0|true_gen=0/1])": -np.log10(error_estimates[error_to_index[(1, 0)]]),
	"-log10(P[obs=0/1|true_gen=0/1])": -np.log10(baseline[1]),
	"-log10(P[obs=1/1|true_gen=0/1])": -np.log10(error_estimates[error_to_index[(1, 2)]]),

	"-log10(P[obs=./.|true_gen=1/1])": -np.log10(error_estimates[error_to_index[(2, 3)]]),
	"-log10(P[obs=0/0|true_gen=1/1])": -np.log10(error_estimates[error_to_index[(2, 0)]]),
	"-log10(P[obs=0/1|true_gen=1/1])": -np.log10(error_estimates[error_to_index[(2, 1)]]),
	"-log10(P[obs=1/1|true_gen=1/1])": -np.log10(baseline[2]),

	"-log10(P[obs=./.|true_gen=-/0])": -np.log10(error_estimates[error_to_index[(4, 3)]]),
	"-log10(P[obs=0/0|true_gen=-/0])": -np.log10(baseline[4]),
	"-log10(P[obs=0/1|true_gen=-/0])": -np.log10(error_estimates[error_to_index[(4, 1)]]),
	"-log10(P[obs=1/1|true_gen=-/0])": -np.log10(error_estimates[error_to_index[(4, 2)]]),

	"-log10(P[obs=./.|true_gen=-/1])": -np.log10(error_estimates[error_to_index[(5, 3)]]),
	"-log10(P[obs=0/0|true_gen=-/1])": -np.log10(error_estimates[error_to_index[(5, 0)]]),
	"-log10(P[obs=0/1|true_gen=-/1])": -np.log10(error_estimates[error_to_index[(5, 1)]]),
	"-log10(P[obs=1/1|true_gen=-/1])": -np.log10(baseline[5]),

	"-log10(P[obs=./.|true_gen=-/-])": -np.log10(baseline[6]),
	"-log10(P[obs=0/0|true_gen=-/-])": -np.log10(error_estimates[error_to_index[(6, 0)]]),
	"-log10(P[obs=0/1|true_gen=-/-])": -np.log10(error_estimates[error_to_index[(6, 1)]]),
	"-log10(P[obs=1/1|true_gen=-/-])": -np.log10(error_estimates[error_to_index[(6, 2)]]),

	"x-times higher probability of error in hard-to-sequence region": 10
}

with open(out_file, 'w+') as f:
	json.dump(params, f, indent=4)

