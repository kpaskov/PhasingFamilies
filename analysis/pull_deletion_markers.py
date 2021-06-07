from collections import defaultdict, namedtuple
import sys
import json
from input_output import pull_families, pull_gen_data_for_individuals

phase_dir = sys.argv[1]

with open('%s/info.json' % phase_dir, 'r') as f:
	data_dir = json.load(f)['data_dir']
	assembly = json.load(f)['assembly']
	ped_file = json.load(f)['ped_file']


with open('%s/deletions.json' % phase_dir, 'r') as f:
	deletions = json.load(f)

families = pull_families_missing_parent(ped_file, data_dir)
famkey_to_family = dict([(x.id, x) for x in families])

for chrom in [str(x) for x in range(1, 23)]:
	chrom_dels = [x for x in deletions if x['chrom']==chrom]

	for d in chrom_dels:
		individuals = famkey_to_family[d.family].individuals
		family_genotypes, family_snp_positions, mult_factor = pull_gen_data_for_individuals(data_dir, assembly, chrom, individuals, start_pos=d['start_pos'], end_pos=d['end_pos'])

		print(family_genotypes.shape)
