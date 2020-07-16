import numpy as np
import argparse
import json
from pysam import VariantFile, TabixFile

parser = argparse.ArgumentParser(description='Pull allele frequency from gnomad.')
parser.add_argument('data_dir', type=str, help='Data directory.')
parser.add_argument('gnomad_vcf_file', type=str, help='Gnomad VCF file to pull from allele frequency from.')
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
			af = float(af[0])
			break
	return af

def pull_af_from_gnomad(records):
	positions = []
	afs = []
	for line in records:
		pieces = line.strip().split('\t')
		info = pieces[7].strip().split(';')
		positions.append(int(pieces[1]))
		is_snp = 'variant_type=SNV' in info
		if is_snp:
			positions.append(int(pieces[1]))
			afs.append(pull_af_from_info(info))
			
	return np.array(positions), np.array(afs)

if args.batch_num < num_batches:
	# pull positions of interest
	coord_file = '%s/chr.%s.%d.gen.coordinates.npy' % (args.data_dir, args.chrom, args.batch_num)
	pos_data = np.load(coord_file)
	if pos_data.shape[0]>0:
		is_snp = pos_data[:, 2].astype(bool)
		is_pass = pos_data[:, 3].astype(bool)
	assert np.all(pos_data[is_snp][1:, 1]>pos_data[is_snp][:-1, 1])

	# load AF from appropriate section of gnomad vcf_file
	vcf = TabixFile(args.gnomad_vcf_file, parser=None)
	if batch_size is not None:
		start_pos, end_pos = args.batch_num*batch_size, (args.batch_num+1)*batch_size
		print('Interval', start_pos, end_pos)
		gnomad_positions, gnomad_afs = pull_af_from_gnomad(vcf.fetch(reference='chr%s' % chrom, start=start_pos, end=end_pos))
	else:
		gnomad_positions, gnomad_afs = pull_af_from_gnomadprocess_body(vcf.fetch(reference=contig.name))
	assert np.all(gnomad_positions[1:]>gnomad_positions[:-1])

	# pull AF for positions of interest
	af = -np.ones((pos_data.shape[0],))

	data_indices = is_snp & np.isin(pos_data[:, 1], positions)
	gnomad_indices = np.isin(positions, pos_data[:, 1])

	af[data_indices] = gnomad_afs[gnomad_indices]
	af = np.clip(af, 3/(2*71702), 1-(3/(2*71702)))
	np.save('%s/chr.%s.%d.gen.af' % (args.data_dir, args.chrom, args.batch_num), af)



