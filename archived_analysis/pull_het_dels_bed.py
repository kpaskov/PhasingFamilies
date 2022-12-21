import gzip
from collections import Counter, defaultdict, namedtuple
import sys
import os

ped_file = sys.argv[1] #'../data/160826.ped'
id_mapping = sys.argv[2] #'../data/160826.iHART.db.query.csv'
variant_dir = sys.argv[3] #'../other_sv_calls/genomeStrip'
out_file = sys.argv[4] # ../other_sv_calls/

# Affection (0=unknown; 1=unaffected; 2=affected)
child_id_to_affected = dict()
child_id_to_sex = dict()
child_id_to_parent_ids = dict()

Deletion = namedtuple('Deletion', ['chrom', 'start', 'end'])

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

sample_id_to_filename = dict()
for filename in os.listdir(variant_dir):
    if filename.endswith(".bed"):
        sample_id = filename.split('.')[0]
        if sample_id in id_switch:
            sample_id_to_filename[id_switch[sample_id]] = os.path.join(variant_dir, filename)
print('Sample IDs', len(sample_id_to_filename))

# load parental deletions
parent_deletions = dict()
for parent_id in sum([list(x) for x in child_id_to_parent_ids.values()], []):
    if parent_id in sample_id_to_filename:
        deletions = []
        with open(sample_id_to_filename[parent_id], 'r') as f:
            for line in f:
                pieces = line.strip().split('\t')
                if pieces[3] == 'DEL':
                    deletions.append(Deletion(pieces[0], int(pieces[1]), int(pieces[2])))
        parent_deletions[parent_id] = deletions
print('Parental deletions loaded')

with open(out_file, 'w+') as outf:
    for child_id, (f_id, m_id) in child_id_to_parent_ids.items():
        if child_id in sample_id_to_filename and f_id in parent_deletions:
            par_deletions = parent_deletions[f_id]
            current_chrom = '1'
            current_chrom_dels = [x for x in par_deletions if x[0] == current_chrom]
            with open(sample_id_to_filename[child_id], 'r') as f:
                for line in f:
                    pieces = line.strip().split('\t')
                    if pieces[3] == 'DEL':
                        child_del = Deletion(pieces[0], int(pieces[1]), int(pieces[2]))
                        if child_del.chrom != current_chrom:
                            current_chrom = pieces[0]
                            current_chrom_dels = [x for x in par_deletions if x.chrom == current_chrom]

                        for d in current_chrom_dels:
                            if d.end < child_del.start:
                                overlap = 0
                            elif child_del.end < d.start:
                                overlap = 0
                                break
                            elif d.start <= child_del.start and d.end >= child_del.end:
                                overlap = child_del.end-child_del.start+1
                            elif child_del.start <= d.start and child_del.end >= d.end:
                                overlap = d.end-d.start+1
                            elif child_del.start < d.end:
                                overlap = d.end - child_del.start + 1
                            elif d.start < child_del.end:
                                overlap = child_del.end - d.start + 1
                            if overlap/(child_del.end-child_del.start+1) >= 0.5 and overlap/(d.end-d.start+1) >= 0.5:
                                outf.write('\t'.join([d.chrom, str(d.start), str(d.end), str(child_del.start), str(child_del.end), '', '', '', '', '%s-%s' % (m_id, child_id)]) + '\n')

        if child_id in sample_id_to_filename and m_id in parent_deletions:
            par_deletions = parent_deletions[m_id]
            current_chrom = '1'
            current_chrom_dels = [x for x in par_deletions if x[0] == current_chrom]
            with open(sample_id_to_filename[child_id], 'r') as f:
                for line in f:
                    pieces = line.strip().split('\t')
                    if pieces[3] == 'DEL':
                        child_del = Deletion(pieces[0], int(pieces[1]), int(pieces[2]))
                        if child_del.chrom != current_chrom:
                            current_chrom = pieces[0]
                            current_chrom_dels = [x for x in par_deletions if x.chrom == current_chrom]

                        for d in current_chrom_dels:
                            if d.end < child_del.start:
                                overlap = 0
                            elif child_del.end < d.start:
                                overlap = 0
                                break
                            elif d.start <= child_del.start and d.end >= child_del.end:
                                overlap = child_del.end-child_del.start+1
                            elif child_del.start <= d.start and child_del.end >= d.end:
                                overlap = d.end-d.start+1
                            elif child_del.start < d.end:
                                overlap = d.end - child_del.start + 1
                            elif d.start < child_del.end:
                                overlap = child_del.end - d.start + 1
                            if overlap/(child_del.end-child_del.start+1) >= 0.5 and overlap/(d.end-d.start+1) >= 0.5:
                                outf.write('\t'.join([d.chrom, str(d.start), str(d.end), str(child_del.start), str(child_del.end), '', '', '', '', '%s-%s' % (m_id, child_id)]) + '\n')



            


            