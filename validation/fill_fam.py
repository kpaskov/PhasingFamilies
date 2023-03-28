import sys

fam_file = sys.argv[1]
ped_file = sys.argv[2]

with open(fam_file, 'r') as f:
	ind_ids = [line.split()[1] for line in f]

with open(ped_file, 'r') as f:
	ind_id_to_line = dict([(line.split()[1], ' '.join(line.split())) for line in f])

with open(fam_file, 'w+') as f:
	for ind_id in ind_ids:
		f.write(ind_id_to_line[ind_id])