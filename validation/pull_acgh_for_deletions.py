import json
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import sys
import gzip
from os import listdir

del_file = sys.argv[1]
acgh_dir = sys.argv[2]

ssc_old_id_to_new_id = dict()
with open('data/ssc.id_map.from.repository', 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        ssc_old_id_to_new_id[pieces[1]] = pieces[0]
        ssc_old_id_to_new_id[pieces[1].replace('.', '_')] = pieces[0]

# pull HybIDs
sampleid_to_hybid = dict()
with open('%s/GSE23682_family_table.txt' % acgh_dir, 'r') as f:
    for line in f:
        # skip header
        if line.startswith('#'):
            pass
        else:
            pieces = line.strip().split('\t')
            sampleid_to_hybid[pieces[1]] = pieces[0]

sampleid_to_datafile = dict()
for f in listdir(acgh_dir):
    if f.endswith('.pubmed.txt.gz'):
        sample_id = [x for x, hyb in sampleid_to_hybid.items() if hyb in f]
        if len(sample_id) == 1:
            sampleid_to_datafile[sample_id[0]] = f
print(len(sampleid_to_datafile))

# pull deletions
with open(del_file) as f:
    deletions = json.load(f)
print(len(deletions))

# pull probe positions
chroms = [str(x) for x in range(1, 23)]
chrom_to_index = dict([(x, i) for i, x in enumerate(chroms)])

probes = []
probe_positions = []
with open('%s/hg19.bed' % acgh_dir) as f:
    for line in f:
        pieces = line.strip().split('\t')
        if pieces[0][3:] in chrom_to_index:
            start_pos, end_pos = int(pieces[1]), int(pieces[2])
            probes.append(pieces[3])
            probe_positions.append((chrom_to_index[pieces[0][3:]], start_pos, end_pos))
probe_positions = np.array(probe_positions)
probe_to_index = dict([(x, i) for i, x in enumerate(probes)])
print(probe_positions.shape)


child_to_deletions = defaultdict(list)
for d in deletions:
    for child in [ssc_old_id_to_new_id.get(x, x) for x in d['trans']]:
        if child in sampleid_to_datafile:
            child_to_deletions[child].append(d)
print(len(child_to_deletions))


for child, childdels in child_to_deletions.items():

    data = np.zeros((len(probes),))
    data.fill(np.nan)
    with gzip.open('%s/%s' % (acgh_dir, sampleid_to_datafile[child]), 'rt') as f:
        # skip header
        line = next(f)
        while line.startswith('#'):
            line = next(f)
                
        for line in f:
            pieces = line.strip().split('\t')
            if pieces[0] in probe_to_index:
                data[probe_to_index[pieces[0]]] = float(pieces[1])

    for d in childdels:
        indices = (probe_positions[:, 0]==chrom_to_index[d['chrom']]) & (probe_positions[:, 1]>=d['start_pos']) & (probe_positions[:, 2]<=d['end_pos']) & (~np.isnan(data))
        #print(data[indices])
        d['%s_num_markers_aCGH' % child] = int(np.sum(indices))
        d['%s_med_aCGH' % child] = float(np.median(data[indices]))

with open(del_file, 'w+') as f:
    json.dump(deletions, f, indent=4)
