import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import product
import scipy.stats
import json

BaselineCounts = namedtuple('BaselineCounts', ['counts', 'samples', 'families', 'family_sizes', 'is_child', 'is_mom', 'is_dad'])
Samples = namedtuple('Samples', ['sample_ids', 'families', 'family_sizes', 'is_child', 'is_mom', 'is_dad'])
PrecisionRecall = namedtuple('PrecisionRecall', ['precision1', 'recall1', 'precision2', 'recall2'])

def pull_samples(data_dir, chroms):
    sample_to_chroms = defaultdict(set)
    sample_to_family = dict()
    family_to_size = dict()
    children, moms, dads = set(), set(), set()
    
    # pull counts from famgen
    for i, chrom in enumerate(chroms):
        print(chrom, end=' ')

        with open('%s/chr.%s.famgen.counts.txt' % (data_dir, chrom), 'r') as f:
            for line in f:
                pieces = line.strip().split('\t')
                famkey, inds = pieces[:2]
                #famkey = famkey.split('.')[0]

                if 'ssc' in data_dir:
                    # unfortunately, ssc uses . in their sample names
                    inds = inds.split('.')
                    inds = ['%s.%s' % (inds[i], inds[i+1]) for i in range(0, len(inds), 2)]
                else:
                    inds = inds.split('.')
                    
                m = len(inds)
                family_to_size[famkey] = m
                for ind in inds:
                    sample_to_chroms[ind].add(chrom)
                    sample_to_family[ind] = famkey

                moms.add(inds[0])
                dads.add(inds[1])
                children.update(inds[2:])    
            
    multigen = children & (moms | dads)
    print('\nRemoving %d individuals involved in multiple generations' % len(multigen))
    
    missing_chroms = set([x for x, chrs in sample_to_chroms.items() if len(chrs) != len(chroms)])
    print('Removing %d individuals missing chromosomal data' % len(missing_chroms))
    
    children = children - multigen - missing_chroms
    moms = moms - multigen - missing_chroms
    dads = dads - multigen - missing_chroms
    
    samples = sorted(children | moms | dads)
    families = [sample_to_family[x] for x in samples]
    family_sizes = np.array([family_to_size[x] for x in families])
    is_child = np.array([x in children for x in samples])
    is_mom = np.array([x in moms for x in samples])
    is_dad = np.array([x in dads for x in samples])
    
    return Samples(samples, families, family_sizes, is_child, is_mom, is_dad)

def pull_baseline_counts(samples, data_dir, chroms, gens, obss):
    sample_to_index = dict([(x, i) for i, x in enumerate(samples.sample_ids)])
    baseline_counts = np.zeros((len(samples.sample_ids), len(obss)))
    
    # pull counts from famgen
    for i, chrom in enumerate(chroms):
        print(chrom, end=' ')

        with open('%s/chr.%s.famgen.counts.txt' % (data_dir, chrom), 'r') as f:
            for line in f:
                pieces = line.strip().split('\t')
                famkey, inds = pieces[:2]
                #famkey = famkey.split('.')[0]

                if 'ssc' in data_dir:
                    # unfortunately, ssc uses . in their sample names
                    inds = inds.split('.')
                    inds = ['%s.%s' % (inds[i], inds[i+1]) for i in range(0, len(inds), 2)]
                else:
                    inds = inds.split('.')
                    
                m = len(inds)
                gen_indices = [x for x in range(m) if inds[x] in sample_to_index]
                family_indices = [sample_to_index[x] for x in inds if x in sample_to_index]

                for g, c in zip(np.array(list(product([0, 1, 2, 3], repeat=m)))[:, gen_indices], pieces[2:]):
                    baseline_counts[family_indices, g] += int(c)
    
    return baseline_counts

def pull_error_rates(samples, param_file, chroms, gens, obss):
    
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    rates = np.zeros((len(samples.sample_ids), len(gens), len(obss)))
    rates[:] = np.nan
                
    for i, (sample_id, family) in enumerate(zip(samples.sample_ids, samples.families)):
        key = '%s.%s' % (family, sample_id)
        if key in params:
            rates[i, :, :] = [[params[key]['-log10(P[obs=%s|true_gen=%s])' % (o, g)] for o in obss] for g in gens]
        
    return rates

def estimate_error_counts(baseline_counts, error_rates, chroms, gens, obss):
    m = error_rates.shape[0]
    expected_error_counts = np.zeros((m, len(gens), len(obss)))
    
    fill_in_values = np.nanmedian(10.0**-error_rates, axis=0).T
    
    for i in range(m):
        ground_truth = baseline_counts[i, :3]
        
        for gen_index, obs_index in product(range(len(gens)), range(len(obss))):
            expected_error_counts[i, gen_index, obs_index] = ground_truth[gen_index]*(10**-error_rates[i, gen_index, obs_index])
    return expected_error_counts

def calculate_precision_recall(error_rates, baseline_counts, gens, obss):
    # precision: TP/(TP + FP)
    # let n_0 = # of times the real genotype is 0/0
    # E[TP] = n_1 * p_11
    # E[FP] = n_0 * p_01

    # figure out true baselines
    ns = baseline_counts[:, :3]

    precisions = []
    recalls = []
    for gen_index, gen in enumerate(gens):
        error_counts = ns*(10.0**-error_rates[:, :, obss.index(gen)])
        TP = error_counts[:, gen_index]
        FP = np.sum(error_counts, axis=1)-TP
        FN = ns[:, gen_index]*(1-(10.00**-error_rates[:, gen_index, obss.index(gen)]))
 
        precisions.append(TP/(TP+FP))
        recalls.append(TP/(TP+FN))
        
    return PrecisionRecall(precisions[1], recalls[1], precisions[2], recalls[2])


