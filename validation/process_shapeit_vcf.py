import gzip
from pysam import TabixFile
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import json
from itertools import islice, groupby

Crossover = namedtuple('Crossover', ['family', 'chrom', 'start_pos', 'end_pos', 'child', 'is_mat', 'is_pat'])

# pull .ped
child_to_mother = dict()
child_to_father = dict()
child_to_family = dict()

with open('../DATA/ssc.hg38/ssc.ped', 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) < 6:
            print('ped parsing error', line)
        else:
            fam_id, child_id, f_id, m_id, _, phen = pieces[0:6]
                
            if f_id != '0':
                child_to_father[child_id] = f_id
            if m_id != '0':
                child_to_mother[child_id] = m_id
            child_to_family[child_id] = fam_id

vcf = TabixFile('data/shapeit/10.phased.vcf.gz', parser=None)

samples = vcf.header[-1].strip().split()[9:]
sample_to_index = dict([(x, i) for i, x in enumerate(samples)])

children = sorted([x for x in child_to_mother.keys() & child_to_father.keys() if x in sample_to_index and \
            child_to_mother[x] in sample_to_index and \
            child_to_father[x] in sample_to_index])
child_indices = np.array([sample_to_index[x] for x in children])
mother_indices = np.array([sample_to_index[child_to_mother[x]] for x in children])
father_indices = np.array([sample_to_index[child_to_father[x]] for x in children])
print(len(children))

with open('data/shapeit/children.json', 'w+') as f:
    json.dump(children, f)

gen_to_map = {'0|0': (0, 0), '1|1': (1, 1), '0|1': (0, 1), '1|0': (1, 0)}

specific_positions_mat = [[] for _ in children]
is_mat1 = [[] for _ in children]

specific_positions_pat = [[] for _ in children]
is_pat1 = [[] for _ in children]


for i, record in enumerate(vcf.fetch()):
    if i%1000==0:
        print(i, end=' ', flush=True)
    pieces = record.strip().split()
    pos = int(pieces[1])
    gens = [gen_to_map[x] for x in pieces[9:]]
    mat_var = np.array([x[1] for x in gens], dtype=int)
    pat_var = np.array([x[0] for x in gens], dtype=int)
    
    is_mat1_match = mat_var[child_indices] == pat_var[mother_indices]
    is_mat2_match = mat_var[child_indices] == mat_var[mother_indices]
    is_pat1_match = pat_var[child_indices] == pat_var[father_indices]
    is_pat2_match = pat_var[child_indices] == mat_var[father_indices]
    
    for i in np.where(is_mat1_match != is_mat2_match)[0]:
        specific_positions_mat[i].append(pos)
        is_mat1[i].append(bool(is_mat1_match[i]))
        
    for i in np.where(is_pat1_match != is_pat2_match)[0]:
        specific_positions_pat[i].append(pos)
        is_pat1[i].append(bool(is_pat1_match[i]))

num_mat_crossovers = np.zeros((len(children), 20), dtype=int)
num_pat_crossovers = np.zeros((len(children), 20), dtype=int)
crossovers = []

for i, child in enumerate(children):
    if i%100==0:
        print(i, end=' ', flush=True)

    # mat crossovers
    num_evidence_per_switch = []
    for _, g in groupby(zip(is_mat1[i], specific_positions_mat[i]), key=lambda x: x[0]):
    	start_pos = next(g)[1]
    	l = 1
    	for x in g:
    		l += 1
    	num_evidence_per_switch.append([l, start_pos, x[1]])
    num_mat_crossovers[i, 0] = len(num_evidence_per_switch)

    for j, evidence_cutoff in enumerate([5, 10, 50, 100]):
        new_num_evidence_per_switch = []
        groups = groupby(num_evidence_per_switch, key=lambda x: x[0]<=evidence_cutoff)
        for k, v in groups:
            if not k:
                new_num_evidence_per_switch.extend(v)
            elif len(new_num_evidence_per_switch)==0:
                pass
            elif len(list(v))%2==1:
                try:
                    _, next_v = next(groups)
                    next_v = list(next_v)
                    new_num_evidence_per_switch[-1][0] += next_v[0][0]
                    new_num_evidence_per_switch[-1][2] = next_v[0][2]
                    new_num_evidence_per_switch.extend(next_v[1:])
                except StopIteration:
                    pass
        num_evidence_per_switch = new_num_evidence_per_switch
        num_mat_crossovers[i, j] = len(num_evidence_per_switch)
        #print(evidence_cutoff, num_evidence_per_switch)
    crossovers.extend([Crossover(child_to_family[child], '10', x[2], y[1], child, True, False) for x, y in \
                        zip(num_evidence_per_switch[:-1], num_evidence_per_switch[1:])])

    # pat crossovers
    num_evidence_per_switch = []
    for _, g in groupby(zip(is_pat1[i], specific_positions_pat[i]), key=lambda x: x[0]):
    	start_pos = next(g)[1]
    	l = 1
    	for x in g:
    		l += 1
    	num_evidence_per_switch.append([l, start_pos, x[1]])
    num_pat_crossovers[i, 0] = len(num_evidence_per_switch)

    for j, evidence_cutoff in enumerate([5, 10, 50, 100]):
        new_num_evidence_per_switch = []
        groups = groupby(num_evidence_per_switch, key=lambda x: x[0]<=evidence_cutoff)
        for k, v in groups:
            if not k:
                new_num_evidence_per_switch.extend(v)
            elif len(new_num_evidence_per_switch)==0:
                pass
            elif len(list(v))%2==1:
                try:
                    _, next_v = next(groups)
                    next_v = list(next_v)
                    new_num_evidence_per_switch[-1][0] += next_v[0][0]
                    new_num_evidence_per_switch[-1][2] = next_v[0][2]
                    new_num_evidence_per_switch.extend(next_v[1:])
                except StopIteration:
                    pass
        num_evidence_per_switch = new_num_evidence_per_switch
        num_pat_crossovers[i, j] = len(num_evidence_per_switch)
    crossovers.extend([Crossover(child_to_family[child], '10', x[2], y[1], child, False, True) for x, y in \
                        zip(num_evidence_per_switch[:-1], num_evidence_per_switch[1:])])

np.save('data/shapeit/num_mat_crossovers.npy', num_mat_crossovers)
np.save('data/shapeit/num_pat_crossovers.npy', num_pat_crossovers)

with open('data/shapeit/crossovers.json', 'w+') as f:
    json.dump(crossovers, f)
