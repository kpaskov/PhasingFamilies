import numpy as np
import argparse
import json
from pysam import VariantFile, TabixFile
import gzip

parser = argparse.ArgumentParser(description='Pull allele frequency from gnomad.')
parser.add_argument('data_dir', type=str, help='Data directory.')
parser.add_argument('gnomad_vcf_file', type=str, help='Gnomad VCF file to pull from allele frequency from.')
parser.add_argument('num_gnomad_samples', type=int, help='Num samples used in this version of gnomad.')
parser.add_argument('chrom', type=str, help='Chromosome of interest.')
parser.add_argument('batch_num', type=int, default=0, help='To be used to restrict positions per file. Will include positions >= batch_num*batch_size and <= (batch_num+1)*batch_size')
args = parser.parse_args()


# pull metadata
with open('%s/info.json' % args.data_dir, 'r') as f:
	info = json.load(f)
	batch_size = info['batch_size']
	assembly = info['assembly']

with open('data/chrom_lengths%s.json' % assembly, 'r') as f:
	chrom_length = json.load(f)[args.chrom]

# calculate how many batches should exist
if batch_size is None:
	num_batches = 1
else:
	num_batches = np.ceil(chrom_length/batch_size)

def pull_af_from_info(info):
	af = -1
	for entry in info:
		if entry.startswith('AF='):
			af = float(entry[3:])
			break
	return af

def pull_af_from_gnomad(records, positions, refs, alts, n):
	position_to_index = dict([(x, i) for i, x in enumerate(positions)])
	afs = np.zeros((len(positions),))
	for line in records:
		pieces = line.strip().split('\t')
		info = pieces[7].strip().split(';')
		if (int(pieces[1]) in position_to_index):
			index = position_to_index[int(pieces[1])]
			if (refs[index] == pieces[3]) and (alts[index] == pieces[4]):
				afs[index] = pull_af_from_info(info)
	return np.clip(np.array(afs), 3/(2*n), 1-(3/(2*n))) # rule of 3

if args.batch_num < num_batches:
	# pull positions of interest
	coord_file = '%s/chr.%s.%d.gen.coordinates.npy' % (args.data_dir, args.chrom, args.batch_num)
	pos_data = np.load(coord_file)
	if pos_data.shape[0]>0:
		positions = pos_data[:, 1]
	else:
		positions = np.zeros((0,), dtype=int)
	print(positions.shape)

	# pull ref/alt alleles
	refs, alts = [], []
	with gzip.open('%s/chr.%s.%d.gen.variants.txt.gz' % (args.data_dir, args.chrom, args.batch_num), 'rt') as f:
		for line in f:
			pieces = line.strip().split('\t', maxsplit=5)
			refs.append(pieces[3])
			alts.append(pieces[4])
	print(len(refs), len(alts))

	if positions.shape[0]>0:
		vcf = TabixFile(args.gnomad_vcf_file, parser=None)
		if batch_size is not None:
			start_pos, end_pos = args.batch_num*batch_size, (args.batch_num+1)*batch_size
			print('Interval', start_pos, end_pos)
			gnomad_afs = pull_af_from_gnomad(vcf.fetch(reference='chr%s' % args.chrom, start=start_pos, end=end_pos),
				positions, refs, alts, args.num_gnomad_samples)
		else:
			gnomad_afs = pull_af_from_gnomad(vcf.fetch(reference='chr%s' % args.chrom),
				positions, refs, alts, args.num_gnomad_samples)
	else:
		gnomad_afs = np.zeros((0,))

	np.save('%s/chr.%s.%d.gen.af' % (args.data_dir, args.chrom, args.batch_num), gnomad_afs)



