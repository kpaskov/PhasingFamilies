from collections import defaultdict, namedtuple
import sys
import json
import numpy as np
from input_output import pull_families, chrom_lengths37, chrom_lengths38
from os import listdir
from scipy import sparse

data_dir = sys.argv[1]
phase_dir = sys.argv[2]
marker_name = sys.argv[3]

with open('%s/info.json' % phase_dir, 'r') as f:
	phase_info = json.load(f)
	ped_file = phase_info['ped_file']

	if 'assembly' in phase_info:
		phase_assembly = phase_info['assembly']
	else:
		phase_data_dir = phase_info['data_dir']
		with open('%s/info.json' % phase_data_dir, 'r') as f:
			phase_assembly = json.load(f)['assembly']
	
with open('%s/info.json' % data_dir, 'r') as f:
	data_info = json.load(f)
	assert phase_assembly == data_info['assembly']
	assembly = data_info['assembly']

with open('%s/deletions.json' % phase_dir, 'r') as f:
	deletions = json.load(f)

families = pull_families(ped_file, data_dir)
famkey_to_family = dict([(x.id, x) for x in families])

print(data_dir, assembly)

# pull samples
sample_file = '%s/samples.json' % data_dir
with open(sample_file, 'r') as f:
	sample_ids = json.load(f)
sample_id_to_index = dict([(sample_id, i) for i, sample_id in enumerate(sample_ids)])


for chrom in [str(x) for x in range(1, 23)]:
	print(chrom)
	# pull gen data
	gen_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.npz' in f], key=lambda x: int(x.split('.')[2]))
	coord_files = sorted([f for f in listdir(data_dir) if ('chr.%s.' % chrom) in f and 'gen.coordinates.npy' in f], key=lambda x: int(x.split('.')[2]))
	assert len(gen_files) == len(coord_files)

	# pull chrom length
	assert assembly == '37' or assembly == '38'
	chrom_length = chrom_lengths37[chrom] if assembly == '37' else chrom_lengths38[chrom] if assembly == '38' else None

	gens, snp_positions = [], []
	for gen_file, coord_file in zip(gen_files, coord_files):
		coords = np.load('%s/%s' % (data_dir, coord_file))

		if coords.shape[0]>0:
			poss = coords[:, 1]
			is_snp = coords[:, 2]==1
			is_pass = coords[:, 3]==1

			has_data = is_snp & is_pass
			if np.sum(has_data)>0:
				gen = sparse.load_npz('%s/%s' % (data_dir, gen_file))

				
				gens.append(gen[:, has_data].A)
				snp_positions.append(poss[has_data])

	gens = np.hstack(gens)
	snp_positions = np.hstack(snp_positions)
	position_to_index = dict([(x, i) for i, x in enumerate(snp_positions)])
	assert np.all(snp_positions <= chrom_length)

	# remove multiallelic sites
	is_multiallelic = np.zeros((snp_positions.shape[0],), dtype=bool)
	indices = np.where(snp_positions[:-1] == snp_positions[1:])[0]
	is_multiallelic[indices] = True
	is_multiallelic[indices+1] = True
	#print(np.sum(is_multiallelic))

	for d in deletions:
		if (d['chrom'] == chrom):
			individuals = famkey_to_family[d['family']].individuals
			ind_indices = [sample_id_to_index[x] for x in individuals if x in sample_id_to_index] 

			if len(ind_indices)==0:
				d[marker_name] = 0
			else:
				in_interval = (snp_positions>=d['start_pos']) & (snp_positions<=d['end_pos'])

				if np.sum(~is_multiallelic & in_interval)==0:
					d[marker_name] = 0
				else:
					d[marker_name] = int(np.sum(np.any(gens[np.ix_(ind_indices, ~is_multiallelic & in_interval)]>0, axis=0)))

with open('%s/deletions.json' % phase_dir, 'w+') as f:
	json.dump(deletions, f, indent=4)

	