import gzip
from collections import Counter
import sys

ped_file = sys.argv[1] #'../data/160826.ped'
id_mapping = sys.argv[2] #'../data/160826.iHART.db.query.csv'
variant_file = sys.argv[3] #'../other_sv_calls/bd.b12_19.vcf.gz'
out_file = sys.argv[4] # ../other_sv_calls/

# Affection (0=unknown; 1=unaffected; 2=affected)
child_id_to_affected = dict()
child_id_to_sex = dict()
child_id_to_parent_ids = dict()

with open(ped_file, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) >= 6:
            fam_id, child_id, f_id, m_id, sex, disease_status = pieces[0:6]
            child_id_to_affected[child_id] = disease_status
            child_id_to_sex[child_id] = sex
            child_id_to_parent_ids[child_id] = (f_id, m_id)

id_switch = dict()
with open(id_mapping, 'r') as f:
    for line in f:
        pieces = line.strip().split(',')
        id_switch[pieces[1]] = pieces[2]

with gzip.open(variant_file, 'rb') as f, open(out_file, 'w+') as outf:
    line = next(f).decode("utf-8").strip()
    while line.startswith('##'):
        line = next(f).decode("utf-8").strip()
    
    sample_ids = [id_switch[x] for x in line.strip().split('\t')[9:]]
    
    # pull sample ids
    for line in f:
        pieces = line.decode("utf-8").strip().split('\t', maxsplit=5)
        if pieces[4] == '<DEL>':
            pieces = pieces[:-1] + pieces[-1].split('\t')
            hets = [x for x, y in zip(sample_ids, pieces[9:]) if y[:3]=='0/1']
            if len(hets)>1:
            	outf.write('\t'.join(pieces[:9]+hets) + '\n')



            