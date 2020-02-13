import os
import sys
from collections import defaultdict

phase_dir = sys.argv[1]

# families files
file_to_batches = defaultdict(list)
for filename in os.listdir(phase_dir):
	if 'families' in filename:
		batch_start, batch_end = filename.index('families')+8, filename.index('.txt')
		batch_num = filename[batch_start:batch_end]
		if batch_num != '':
			batch_num = int(batch_num)
			file_to_batches[os.path.join(phase_dir, filename[:batch_start] + filename[batch_end:])].append((batch_num, os.path.join(phase_dir, filename)))

for filename, batches in file_to_batches.items():
	print(filename)
	batches = sorted(batches, key=lambda x: x[0])

	try:
		with open(filename, 'w+') as f:
			for i, (_, subfile) in enumerate(batches):
				with open(subfile, 'r') as sf:
					if i!=0:
						next(sf) # skip header on subsequent files
					for line in sf:
						f.write(line)
	except:
		print('Error', filename)

# phased files
file_to_batches = defaultdict(list)
for filename in os.listdir(phase_dir):
	if 'phased' in filename and 'masked' not in filename:
		batch_start, batch_end = filename.index('phased')+6, filename.index('.txt')
		batch_num = filename[batch_start:batch_end]
		if batch_num != '':
			batch_num = int(batch_num)
			file_to_batches[os.path.join(phase_dir, filename[:batch_start] + filename[batch_end:])].append((batch_num, os.path.join(phase_dir, filename)))


for filename, batches in file_to_batches.items():
	print(filename)
	batches = sorted(batches, key=lambda x: x[0])

	try:
		with open(filename, 'w+') as f:
			for i, (_, subfile) in enumerate(batches):
				with open(subfile, 'r') as sf:
					if i!=0:
						next(sf) # skip header on subsequent files
					for line in sf:
						f.write(line)
	except:
		print('Error', filename)

