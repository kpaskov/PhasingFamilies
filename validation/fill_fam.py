import sys

fam_file = sys.argv[1]
ped_file = sys.argv[2]

with open(fam_file, 'r') as f:
	lines = [line for line in f]

with open(ped_file, 'r') as f:
	ind_id_to_line = dict([(line.split()[1], ' '.join(line.split())+'\n') for line in f])

with open(fam_file, 'w+') as f:
	for line in lines:
		ind_id = line.split()[1]
		if ind_id in ind_id_to_line:
			f.write(ind_id_to_line[ind_id])
		else:
			f.write(line)