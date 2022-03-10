import json
from collections import defaultdict, Counter
import numpy as np
from os import listdir
import scipy.stats as stats
import gzip
import scipy.sparse as sparse
import statsmodels.api as sm
import random
from sklearn.metrics import precision_recall_curve, roc_curve
from itertools import combinations, product
import csv
import cvxpy as cp
import sys

ped_file = '../DATA/ssc.hg38/ssc.ped'
phase_dir = 'recomb_ssc.hg38'
lamb = float(sys.argv[1])
scq_index = int(sys.argv[2])

colors=['#ef6c00', '#4db6ac', '#ce93d8ff']

# pull affected status
# (0=unknown; 1=unaffected; 2=affected)
child_id_to_affected = dict()
child_id_to_sex = dict()
with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) >= 6:
            fam_id, child_id, f_id, m_id, sex, disease_status = pieces[0:6]
            if 'ssc' in ped_file:
                child_id = child_id.replace('.', '_')
            child_id_to_affected[child_id] = disease_status
            child_id_to_sex[child_id] = sex


print('ped loaded')
print('children', len(child_id_to_affected))


with open('%s/sibpairs.json' % phase_dir) as f:
    sibpairs = sorted([(x['family'], x['sibling1'], x['sibling2']) for x in json.load(f)])
sibpair_to_index = dict([(x, i) for i, x in enumerate(sibpairs)])

sample_to_phen = dict()
with open('phenotypes/ssc/designated.unaffected.sibling.data/scq_life_raw.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    for pieces in reader:
        sample_to_phen[pieces[0]] = pieces[2:]
        
with open('phenotypes/ssc/proband.data/scq_life_raw.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    for pieces in reader:
        sample_to_phen[pieces[0]] = pieces[2:]

aut_response = ['no', 'no', 'yes', 'yes', 'yes',
                'yes', 'yes', 'yes', 'no', 'yes',
                'yes', 'yes', 'yes', 'yes', 'yes',
                'yes', 'yes', 'yes', 'no', 'no',
                'no', 'no', 'no', 'no', 'no',
                'no', 'no', 'no', 'no', 'no',
                'no', 'no', 'no', 'no', 'no',
                'no', 'no', 'no', 'no', 'no']

phen = np.zeros((len(sibpairs), 40), dtype=int)
is_missing = np.zeros((len(sibpairs), 40), dtype=bool)

for i, sibpair in enumerate(sibpairs):
    if (sibpair[0]+'.p1') in sample_to_phen and (sibpair[0]+'.s1') in sample_to_phen:
        sib1_phen = sample_to_phen[sibpair[0] + ('.p1' if child_id_to_affected[sibpair[1]]=='2' else '.s1')]
        sib2_phen = sample_to_phen[sibpair[0] + ('.p1' if child_id_to_affected[sibpair[2]]=='2' else '.s1')]

        for j in range(40):
            if (sib1_phen[j] in {'yes', 'no'}) and (sib2_phen[j] in {'yes', 'no'}):
                phen[i, j] = (int(sib1_phen[j]==aut_response[j]) - int(sib2_phen[j]==aut_response[j]))
            else:
                is_missing[i, j] = True
    else:
        is_missing[i, :] = True

with open('%s/deletions.json' % phase_dir) as f:
    deletions = json.load(f)

# filter deletions
deletions = [d for d in deletions if d['length']>=1000]
print('remaining deletions', len(deletions))

deletions = [d for d in deletions if d['is_inherited']]
print('remaining deletions', len(deletions))
    
#deletions = [d for d in deletions if not d['is_hts']]
#print('remaining deletions', len(deletions))

X = []
positions = []
for chrom in [str(x) for x in range(1, 23)] + ['X']:
    chrom_ds = [d for d in deletions if d['chrom'] == chrom]
    poss = sorted(set([d['start_pos']-1 for d in chrom_ds if len(d['trans'])==1 and len(d['notrans'])==1]) | \
                  set([d['start_pos'] for d in chrom_ds if len(d['trans'])==1 and len(d['notrans'])==1]) | \
                  set([d['end_pos']-1 for d in chrom_ds if len(d['trans'])==1 and len(d['notrans'])==1]) | \
                  set([d['end_pos'] for d in chrom_ds if len(d['trans'])==1 and len(d['notrans'])==1]))
    
    pos_to_index = dict([(x, i) for i, x in enumerate(poss)])
    
    dels = np.zeros((len(sibpairs), len(poss)), dtype=int)
    for d in chrom_ds:
        if len(d['trans'])==1 and len(d['notrans'])==1:
            start_index = pos_to_index[d['start_pos']]
            end_index = pos_to_index[d['end_pos']]

            sibs = d['trans'] + d['notrans']
            if (d['family'], sibs[0], sibs[1]) not in sibpair_to_index:
                sibs = [sibs[1], sibs[0]]

            sibpair_index = sibpair_to_index[(d['family'], sibs[0], sibs[1])]

            if d['trans'][0]==sibs[0]:
                dels[sibpair_index, start_index:end_index] = 1
            else:
                dels[sibpair_index, start_index:end_index] = -1
    
    X.append(dels)
    positions.extend([(chrom, p) for p in poss])
X = np.hstack(X)
print(X.shape)
    

is_mm = np.array([(child_id_to_sex[sib1]=='1') and (child_id_to_sex[sib2]=='1') for (fam, sib1, sib2) in sibpairs])
is_mf = np.array([(child_id_to_sex[sib1]=='1') and (child_id_to_sex[sib2]=='2') for (fam, sib1, sib2) in sibpairs])
is_fm = np.array([(child_id_to_sex[sib1]=='2') and (child_id_to_sex[sib2]=='1') for (fam, sib1, sib2) in sibpairs])
is_ff = np.array([(child_id_to_sex[sib1]=='2') and (child_id_to_sex[sib2]=='2') for (fam, sib1, sib2) in sibpairs])

#is_ntaff = np.array([(child_id_to_affected[sib1]=='1') and (child_id_to_affected[sib2]=='2') for (fam, sib1, sib2) in sibpairs])
#is_affnt = np.array([(child_id_to_affected[sib1]=='2') and (child_id_to_affected[sib2]=='1') for (fam, sib1, sib2) in sibpairs])

is_ntaff = phen[:, scq_index]==-1
is_affnt = phen[:, scq_index]==1


X = np.hstack((is_mm[:, np.newaxis], is_mf[:, np.newaxis], is_fm[:, np.newaxis], is_ff[:, np.newaxis], X))


print(np.sum(X[:, 4:])/(X.shape[0]*(X.shape[1]-4)))
print(np.sum(X[:, 0])/X.shape[0])
print(np.sum(X[:, 1])/X.shape[0])
print(np.sum(X[:, 2])/X.shape[0])
print(np.sum(X[:, 3])/X.shape[0])
print(np.sum(is_affnt)/len(sibpairs), np.sum(is_ntaff)/len(sibpairs))

beta = cp.Variable(X.shape[1])
log_likelihood = cp.sum(
    cp.multiply(is_affnt[is_affnt | is_ntaff], X[is_affnt | is_ntaff, :] @ beta) - cp.logistic(X[is_affnt | is_ntaff, :] @ beta)
)
problem = cp.Problem(cp.Maximize(log_likelihood/np.sum(is_affnt | is_ntaff) - lamb*cp.tv(beta[2:])),
                    [beta[2:]>=0])
#- 0.001*cp.norm(beta[2:], 1) 
problem.solve(solver='MOSEK', verbose=True)

np.save('%s/delmodel.%0.2f.%d' % (phase_dir, lamb, scq_index), beta.value)

