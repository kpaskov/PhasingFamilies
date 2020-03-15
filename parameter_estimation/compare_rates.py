import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import product
import scipy.stats
import json

BaselineCounts = namedtuple('BaselineCounts', ['counts', 'samples', 'families', 'family_sizes', 'is_child', 'is_mom', 'is_dad'])
Samples = namedtuple('Samples', ['sample_ids', 'families', 'family_sizes', 'is_child', 'is_mom', 'is_dad'])
PrecisionRecall = namedtuple('PrecisionRecall', ['precision1', 'recall1', 'precision2', 'recall2',
                    'precision1_upper_bound', 'recall1_upper_bound', 'precision2_upper_bound', 'recall2_upper_bound'])

def pull_samples(data_dir, chroms, dot_in_name=False):
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

                if dot_in_name:
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

def pull_error_rates(samples, param_file, gens, obss):
    
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    rates = np.zeros((len(samples.sample_ids), len(gens), len(obss)))
    rates[:] = np.nan
                
    for i, (sample_id, family) in enumerate(zip(samples.sample_ids, samples.families)):
        key = '%s.%s' % (family, sample_id)
        if key in params:
            rates[i, :, :] = [[params[key]['-log10(P[obs=%s|true_gen=%s])' % (o, g)] for o in obss] for g in gens]
        
    return rates

def pull_error_counts(samples, param_file, gens, obss):

    with open(param_file, 'r') as f:
        params = json.load(f)
    
    counts = np.zeros((len(samples.sample_ids), len(gens), len(obss)))
    counts[:] = np.nan
                
    for i, (sample_id, family) in enumerate(zip(samples.sample_ids, samples.families)):
        key = '%s.%s' % (family, sample_id)
        if key in params:
            counts[i, :, :] = [[params[key]['E[obs=%s, true_gen=%s]' % (o, g)] for o in obss] for g in gens]
        
    return counts

def pull_num_sites(samples, param_file):

    with open(param_file, 'r') as f:
        params = json.load(f)
    
    num_sites = np.zeros((len(samples.sample_ids),))
    num_sites[:] = np.nan
                
    for i, (sample_id, family) in enumerate(zip(samples.sample_ids, samples.families)):
        key = '%s.%s' % (family, sample_id)
        if key in params:
            num_sites[i] = sum([params[key]['observed_%s' % g] for g in ['0/0', '0/1', '1/1']])
        
    return num_sites

def pull_precision_recall(samples, param_file):
    with open(param_file, 'r') as f:
        params = json.load(f)

    het_precision = np.zeros((len(samples.sample_ids),))
    het_precision[:] = np.nan
    het_recall = np.zeros((len(samples.sample_ids),))
    het_recall[:] = np.nan

    homalt_precision = np.zeros((len(samples.sample_ids),))
    homalt_precision[:] = np.nan
    homalt_recall = np.zeros((len(samples.sample_ids),))
    homalt_recall[:] = np.nan

    het_precision_upper_bound = np.zeros((len(samples.sample_ids),))
    het_precision_upper_bound[:] = np.nan
    het_recall_upper_bound = np.zeros((len(samples.sample_ids),))
    het_recall_upper_bound[:] = np.nan

    homalt_precision_upper_bound = np.zeros((len(samples.sample_ids),))
    homalt_precision_upper_bound[:] = np.nan
    homalt_recall_upper_bound = np.zeros((len(samples.sample_ids),))
    homalt_recall_upper_bound[:] = np.nan
                
    for i, (sample_id, family) in enumerate(zip(samples.sample_ids, samples.families)):
        key = '%s.%s' % (family, sample_id)
        if key in params:
            het_precision[i] = params[key]['precision_0/1']
            het_recall[i] = params[key]['recall_0/1']
            homalt_precision[i] = params[key]['precision_1/1']
            homalt_recall[i] = params[key]['recall_1/1']

            het_precision_upper_bound[i] = params[key]['upper_bound[precision_0/1]']
            het_recall_upper_bound[i] = params[key]['upper_bound[recall_0/1]']
            homalt_precision_upper_bound[i] = params[key]['upper_bound[precision_1/1]']
            homalt_recall_upper_bound[i] = params[key]['upper_bound[recall_1/1]']
        
    return PrecisionRecall(het_precision, het_recall, homalt_precision, homalt_recall,
        het_precision_upper_bound, het_recall_upper_bound, homalt_precision_upper_bound, homalt_recall_upper_bound)

