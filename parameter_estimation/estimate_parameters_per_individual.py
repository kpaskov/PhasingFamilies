import sys
import numpy as np
import scipy.stats
from itertools import product
import cvxpy as cp
from collections import Counter, defaultdict
import json

import argparse

parser = argparse.ArgumentParser(description='Estimate parameters.')
parser.add_argument('data_dir', type=str, help='Directory of genotype data in .npy format.')
parser.add_argument('out_file', type=str, help='Output file.')
parser.add_argument('--total_sites', type=int, default=None, help='The total number of sites genotyped. We assume that any sites not included in the counts files are homozygous reference for all individuals.')
parser.add_argument('--is_ngs', action='store_true', default=False, help='True if this data is NGS. The important point is whether or not sites where all individuals are homozygous reference are sometimes dropped from the VCF. If this happens, use flag --is_ngs')
parser.add_argument('--sample_names_have_period', action='store_true', default=False, help='If sample names include periods, we have to do a special parse.')
args = parser.parse_args()

# ------------------------------------ Basic Info ------------------------------------
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


chroms = [str(x) for x in range(1, 23)] #+ ['X', 'Y'] 

# 0 = 0/0
# 1 = 0/1
# 2 = 1/1
# 3 = ./.

gens = ['0/0', '0/1', '1/1']
obss = ['0/0', '0/1', '1/1', './.']

errors = [(0, 1), (0, 2), (0, 3), 
          (1, 0), (1, 2), (1, 3), 
          (2, 0), (2, 1), (2, 3)]
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
allowable_errors_child = {
    (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 3)
}
allowable_errors_parent = {
    (0, 2), (0, 3),
    (1, 0), (1, 2), (1, 3),
    (2, 0), (2, 3)
}


# ------------------------------------ Pull Data ------------------------------------

family_chrom_to_counts = dict()
family_to_inds = dict()
for i, chrom in enumerate(chroms):
    print(chrom, end=' ')
    
    with open('%s/chr.%s.famgen.counts.txt' % (args.data_dir, chrom), 'r') as f:
        for line in f:
            pieces = line.strip().split('\t')
            famkey, inds = pieces[:2]
            
            if args.sample_names_have_period:
            	# unfortunately, ssc uses . in their sample names
                inds = inds.split('.')
                inds = ['%s.%s' % (inds[i], inds[i+1]) for i in range(0, len(inds), 2)]
            else:
                inds = inds.split('.')

            m = len(inds)

            if m<=8:
                if famkey not in family_to_inds:
                    family_to_inds[famkey] = inds
                else:
                    assert family_to_inds[famkey] == inds
                
                counts = np.zeros((4,)*m, dtype=int)
                for g, c in zip(product([0, 1, 2, 3], repeat=m), pieces[2:]):
                    counts[g] = int(c)
                    
                family_chrom_to_counts[(famkey, chrom)] = counts

print('Families of each size', Counter([len(inds) for fkey, inds in family_to_inds.items()]))

# filter families that have all chroms
famkeys = []
for famkey in set([x[0] for x in family_chrom_to_counts.keys()]):
    has_chrom = np.array([(famkey, chrom) in family_chrom_to_counts for chrom in chroms])
    if np.sum(has_chrom) == len(chroms):
        famkeys.append(famkey)
    else:
        print('Missing chromosome counts', famkey, [chroms[i] for i in np.where(~has_chrom)[0]])
famkeys = sorted(famkeys)

## filter families without sex
#famkeys = [k for k in famkeys if np.all([ind in sample_id_to_sex for ind in family_to_inds[k]])]
#print('Families', len(famkeys))

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

def has_variant(x):
    return len([y for y in x if y>0])>0

def estimate_error_rates(is_mendelian, allowable_errors, counts):
    
    # -------------------- set up problem --------------------
    nonmendelian_famgens = [x for x in zip(*np.where(~is_mendelian))]
    print('Mendelian', np.sum(is_mendelian), 'Nonmendelian', len(nonmendelian_famgens))
    
    if args.is_ngs:
        # if we're working with NGS data, we don't know the real counts of famgens without variants
        # because they may have been excluded from the vcf
        nonmendelian_famgens = [x for x in nonmendelian_famgens if has_variant(x)]

    m = len(counts.shape)
    X = np.zeros((len(nonmendelian_famgens), len(errors)*m), dtype=int)
    y = np.zeros((len(nonmendelian_famgens),), dtype=int)

    for k, nmfg in enumerate(nonmendelian_famgens):
        for i, j in product(range(4), range(m)):
            error = (i, nmfg[j])
            if error in allowable_errors[j]:
                neighbor = tuple(i if k==j else nmfg[k] for k in range(m))
                if is_mendelian[neighbor]:
                    error_index = error_to_index[error] + j*len(errors)
                    X[k, error_index] += counts[neighbor]
        y[k] = counts[nmfg]

        
    is_zero = np.sum(X, axis=0)==0
    print('Removing zero cols:', [(np.floor(i/len(errors)), errors[i % len(errors)]) for i in np.where(is_zero)[0]])
    X = X[:, ~is_zero]
    old_col_index_to_new = dict([(old_index, new_index) for new_index, old_index in enumerate(np.where(~is_zero)[0])])

    print('Removing zero rows:', np.sum(np.sum(X, axis=1)==0))
    indices = np.where(np.sum(X, axis=1) != 0)[0]
    X = X[indices, :]
    y = y[indices]
    
    print(X.shape, y.shape)
    
    # -------------------- solve problem --------------------
    
    print('Estimating...', X.shape, y.shape)
    alpha = 1.0/np.max(X)
    
    # cvxpy
    n = cp.Variable(X.shape[1])
    mu = np.sum(X, axis=0)
    objective = cp.Minimize(alpha*mu*n - alpha*y*cp.log(X*n))

    # Wilson score interval so that if we don't observe any errors, then we take the 95% confidence interval
    z = 1.96
    lower_bound = ((z*z)/2)/(mu+(z*z))
    prob = cp.Problem(objective, [n >= lower_bound, n<=1])
    
    result = prob.solve(solver='ECOS', max_iters=10000)
    print(prob.status)
    
    #print(n.value, n.value.shape)
    ns = np.asarray([v for v in n.value])
    
    if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
        raise Error('Parameters not fully estimated.')
        
    # -------------------- reformat solution --------------------

    error_rates = np.zeros((len(inds), len(gens), len(obss)), dtype=float)
    lower_bounds = np.zeros((len(inds), len(gens), len(obss)), dtype=float)
    error_rates[:] = np.nan
    lower_bounds[:] = np.nan
    for k in range(len(errors)*len(inds)):
        if k in old_col_index_to_new:
            error = errors[k%len(errors)]
            ind_index = int(np.floor(k/len(errors)))
            error_rates[ind_index, error[0], error[1]] = ns[old_col_index_to_new[k]]
            lower_bounds[ind_index, error[0], error[1]] = lower_bound[old_col_index_to_new[k]]

    # now fill in P(obs=true_gen)
    for i, gen in enumerate(gens):
        error_rates[:, i, i] = 1-np.sum(error_rates[:, i, [k for k in range(len(obss)) if k != i]], axis=1)
    
    return error_rates, lower_bounds        




## ------------------------------------ Calculate Various Metrics ------------------------------------

def add_observed_counts(params, counts, j, m):
    params['observed_0/0'] = int(np.sum(counts[tuple(0 if x==j else slice(None, None, None) for x in range(m))]))
    params['observed_0/1'] = int(np.sum(counts[tuple(1 if x==j else slice(None, None, None) for x in range(m))]))
    params['observed_1/1'] = int(np.sum(counts[tuple(2 if x==j else slice(None, None, None) for x in range(m))]))
    params['observed_./.'] = int(np.sum(counts[tuple(3 if x==j else slice(None, None, None) for x in range(m))]))

def add_estimated_error_rates(params, error_rates, lower_bounds, j):
    for gen_index, gen in enumerate(gens):
        for obs_index, obs in enumerate(obss):
            params['-log10(P[obs=%s|true_gen=%s])' % (obs, gen)] = float(-np.log10(error_rates[j, gen_index, obs_index]))
            params['lower_bound[-log10(P[obs=%s|true_gen=%s])]' % (obs, gen)] = float(-np.log10(lower_bounds[j, gen_index, obs_index]))

def add_expected_counts(params):
    # we assume error rates are low, so the number of times we observe a genotype is a good estimate of the number of times this genotype actually occurs.
    for gen_index, gen in enumerate(gens):
        for obs_index, obs in enumerate(obss):
            params['E[obs=%s, true_gen=%s]' % (obs, gen)] = params['observed_%s' % gen] * (10.0**-params['-log10(P[obs=%s|true_gen=%s])' % (obs, gen)])
            params['lower_bound[E[obs=%s, true_gen=%s]]' % (obs, gen)] = params['observed_%s' % gen] * (10.0**-params['lower_bound[-log10(P[obs=%s|true_gen=%s])]' % (obs, gen)])

def add_precision_recall(params):
    # precision: TP/(TP + FP)
    # let n_0 = # of times the real genotype is 0/0
    # E[TP] = n_1 * p_11
    # E[FP] = n_0 * p_01

    # we again assume error rates are low, so the number of times we observe a genotype is a good estimate of the number of times this genotype actually occurs.

    for var in gens:
        TP = params['E[obs=%s, true_gen=%s]' % (var, var)]
        FP = np.sum([params['E[obs=%s, true_gen=%s]' % (var, gen)] for gen in gens if var != gen])
        FN = np.sum([params['E[obs=%s, true_gen=%s]' % (obs, var)] for obs in obss if var != obs])

        FP_lb = np.sum([params['lower_bound[E[obs=%s, true_gen=%s]]' % (var, gen)] for gen in gens if var != gen])
        FN_lb = np.sum([params['lower_bound[E[obs=%s, true_gen=%s]]' % (obs, var)] for obs in obss if var != obs])

        params['precision_%s' % var] = TP/(TP+FP)
        params['recall_%s' % var] = TP/(TP+FN)

        params['upper_bound[precision_%s]' % var] = TP/(TP+FP_lb)
        params['upper_bound[recall_%s]' % var] = TP/(TP+FN_lb)

# ------------------------------------ Estimate Error Rates ------------------------------------

params = {}
baseline_match = {(0, 0), (1, 1), (2, 2)}

num_error_families = 0
for i, famkey in enumerate(famkeys):
    print(famkey)
    try:
        inds = family_to_inds[famkey]
        m = len(inds)
            
        is_mendelian = get_mendelian([None, None] + ([mendelian_check]*(m-2)))
        allowable_errors = [allowable_errors_parent]*2 + [allowable_errors_child]*(m-2)
        counts = np.sum(np.array([family_chrom_to_counts[(famkey, chrom)] for chrom in chroms]), axis=0)

        if args.total_sites is not None:
            accounted_for = np.sum(counts)
            if accounted_for > args.total_sites:
                raise Exception('There are more sites in the VCF than you claim. Please adjust --total_sites.')
            counts[(0,)*m] = args.total_sites - accounted_for

        error_rates, lower_bounds = estimate_error_rates(is_mendelian, allowable_errors, counts)

        print(-np.log10(error_rates))

        for j in range(len(inds)):
            # observed counts
            ind_params = {}
            add_observed_counts(ind_params, counts, j, m)
            add_estimated_error_rates(ind_params, error_rates, lower_bounds, j)
            add_expected_counts(ind_params)
            add_precision_recall(ind_params)

            params[famkey + '.' + inds[j]] = ind_params

    except Exception as err:
        num_error_families += 1
        print('ERROR', err)

print('Total errors', num_error_families)

# ------------------------------------ Write to file ------------------------------------

with open(args.out_file, 'w+') as f:
    json.dump(params, f, indent=4)
