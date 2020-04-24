import sys
from itertools import product, compress
from os import listdir
import numpy as np
import scipy.sparse as sparse
import argparse
import gzip

parser = argparse.ArgumentParser(description='Pull family genotype counts. Bins are created representing different combinations of genotype and depth for each family member.')
parser.add_argument('vcf_file', type=str, help='VCF file of variants.')
parser.add_argument('ped_file', type=str, help='Pedigree file (.ped).')
parser.add_argument('chrom', type=str, help='Chromosome.')
parser.add_argument('out_dir', type=str, help='Directory to write counts.')

# parameters involving families
parser.add_argument('--family_sizes', type=int, nargs='+', default=None, help='What size families should we consider?')

# parameters involving variant filters
parser.add_argument('--include', type=str, default=None, help='Regions to include (.bed).')
parser.add_argument('--exclude', type=str, default=None, help='Regions to exclude (.bed).')
parser.add_argument('--filter_pass', action='store_true', default=True, help='Only include variants that PASS.')
parser.add_argument('--filter_snp', action='store_true', default=True, help='Only include variants that are biallelic snps.')
parser.add_argument('--filter_qual', type=float, default=0, help='Only include variants with QUAL greater than cutoff.')

# parameters involving bins
parser.add_argument('--depth_bins', type=int, nargs='+', default=[], 
    help='If counts should be divided into depth bins, give boundaries of the bins. For example, --depth_bins 20 30 divides counts into three depth bins of <20, 20-30, >30.')
parser.add_argument('--genotypes', type=str, nargs='+', default=['0/0', '0/1', '1/1', './.'], 
    help='Genotypes to be considered. If an individual in the family has a genotype not included in this list, the position will be skipped for that family.')

args = parser.parse_args()


# rework non-numeric chromosomes
if args.chrom == '23':
    args.chrom = 'X'
if args.chrom == '24':
    args.chrom = 'Y'
if args.chrom == '25':
    args.chrom = 'MT'

print('filter PASS?', args.filter_pass)
print('filter indels and multiallelic snps?', args.filter_snp)
print('filter QUAL?', args.filter_qual)
print('depth bins', args.depth_bins)
print('genotypes', args.genotypes)

def process_bedfile(bed_file):
    regions = []
    coverage = 0
    with open(bed_file, 'r') as f:
        for line in f:
            if '\t' in line.strip():
                pieces = line.strip().split('\t')
            else:
                pieces = line.strip().split(':')
                pieces = [pieces[0]] + pieces[1].strip().split('-')

            if pieces[0] == args.chrom or pieces[0] == 'chr%s' % args.chrom:
                regions.append(int(pieces[1]))
                regions.append(int(pieces[2])+1)
                coverage += (int(pieces[2])+1 - int(pieces[1]))
    return np.array(regions), coverage

include_regions, include_coverage = (None, 0) if args.include is None else process_bedfile(args.include)
exclude_regions, exclude_coverage = (None, 0) if args.exclude is None else process_bedfile(args.exclude)

print('including %s bp' % ('all' if include_regions is None else str(include_coverage)))
print('excluding %s bp' % ('no' if exclude_regions is None else str(exclude_coverage)))

# pull samples with sequence info
with gzip.open(args.vcf_file, 'rt') as f:
    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)

    sample_ids = line.strip().split('\t')[9:]
    sample_id_to_index = dict([(line.strip(), i) for i, line in enumerate(sample_ids)])
print('individuals with sequence', len(sample_ids))

# pull families from ped file
family_to_indices = dict()
with open(args.ped_file, 'r') as f:	
    for line in f:
        pieces = line.strip().split('\t')
        if len(pieces) < 4:
            print('ped parsing error', line)
        else:
            fam_id, child_id, f_id, m_id = pieces[0:4]

            if child_id in sample_id_to_index and f_id in sample_id_to_index and m_id in sample_id_to_index:
                if (fam_id, m_id, f_id) not in family_to_indices:
                    family_to_indices[(fam_id, m_id, f_id)] = [sample_id_to_index[m_id], sample_id_to_index[f_id]]
                family_to_indices[(fam_id, m_id, f_id)].append(sample_id_to_index[child_id])
families = sorted(family_to_indices.keys())
print('families %d' % len(families))

# update family sizes
if args.family_sizes is None:
    args.family_sizes = sorted(set([len(v) for v in family_to_indices.values()]))
print('family sizes', args.family_sizes)

family_size_to_indices = dict()
family_size_to_counts = dict()

for family_size in args.family_sizes:
    indices = np.array([family_to_indices[k] for k in families if len(family_to_indices[k])==family_size])
    family_size_to_indices[family_size] = indices
    family_size_to_counts[family_size] = np.zeros(tuple([indices.shape[0]] + [len(args.genotypes)]*family_size + [len(args.depth_bins)+1]*family_size), dtype=int)
    print(indices.shape, family_size_to_counts[family_size].shape)

# enumerate all chrom options
chrom_options = [args.chrom, 'chr'+args.chrom]
if args.chrom == 'X':
    chrom_options = chrom_options + ['23', 'chr23']
if args.chrom == 'Y':
    chrom_options = chrom_options + ['24', 'chr24']
if args.chrom == 'MT':
    chrom_options = chrom_options + ['25', 'chr25']

# first, run through VCF file, keeping track of lines that represent positions of interest
positions = []
is_ok_chrom, is_ok_snp, is_ok_pass, is_ok_qual = [], [], [], []
with gzip.open(args.vcf_file, 'rt') as f:
    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)
    next(f)

    for line in f:
        pieces = line.split('\t', maxsplit=2)
        positions.append(pieces[1])

        if pieces[0] in chrom_options:
            pieces = line.strip().split('\t')

            ref, alt = pieces[3:5]
            
            is_ok_chrom.append(True)
            is_ok_snp.append(len(ref) == 1 and len(alt) == 1 and ref != '.' and alt != '.')
            is_ok_pass.append(pieces[6] == 'PASS')
            is_ok_qual.append(float(pieces[5]) >= args.filter_qual)
        else:
            is_ok_chrom.append(False)
            is_ok_snp.append(True)
            is_ok_pass.append(True)
            is_ok_qual.append(True)

positions = np.array(positions)
is_ok_chrom = np.array(is_ok_chrom)
is_ok_snp = np.array(is_ok_snp)
is_ok_pass = np.array(is_ok_pass)
is_ok_qual = np.array(is_ok_qual)

is_ok_variant = is_ok_chrom
if args.filter_snp:
    print('variants excluded due to snp status', np.sum(~is_ok_snp))
    is_ok_variant = is_ok_variant & is_ok_snp
if args.filter_pass:
    print('variants excluded due to pass status', np.sum(~is_ok_pass))
    is_ok_variant = is_ok_variant & is_ok_pass
print('variants excluded due to qual status', np.sum(~is_ok_qual))
is_ok_variant = is_ok_variant & is_ok_qual

# now apply include and exclude filters
if include_regions is not None:
    insert_loc = np.searchsorted(include_regions, positions_of_interest)
    is_ok_include = np.remainder(insert_loc, 2)==1
    print('variants excluded due to include file', np.sum(~is_ok_include))
    is_ok_variant = is_ok_variant & is_ok_include

if exclude_regions is not None:
    insert_loc = np.searchsorted(exclude_regions, positions_of_interest)
    is_ok_exclude = np.remainder(insert_loc, 2)==0
    print('variants excluded due to exclude file', np.sum(~is_ok_exclude))
    is_ok_variant = is_ok_variant & is_ok_exclude

print('final variants of interest', np.sum(is_ok_variant))

# now start pulling genotypes and doing counts
gen_mapping = dict([(x, i) for i, x in enumerate(args.genotypes)])
with gzip.open(args.vcf_file, 'rt') as f:
    # Skip header
    line = next(f)
    while line.startswith('##'):
        line = next(f)
    next(f)

    for j, line in enumerate(compress(f, is_ok_variant)):
        pieces = line.strip().split('\t')
        fmt = pieces[8].strip().split(':')

        # Pull out genotypes and depth
        gen_index = fmt.index('GT')
        dp_index = fmt.index('DP')

        gens = -np.ones((len(sample_ids),), dtype=int)
        dps = -np.ones((len(sample_ids),), dtype=int)

        for i, piece in enumerate(pieces[9:]):
            segment = piece.split(':')
            gens[i] = gen_mapping.get(segment[gen_index], -1) # -1 represents an unknown genotype

            if dp_index < len(segment):
                dp = segment[dp_index]
                if dp.isdigit():
                    dps[i] = int(dp)

        dps = np.digitize(dps, [0] + args.depth_bins)-1 # -1 represents an unknown depth

        for family_size in args.family_sizes:
            indices = family_size_to_indices[family_size]
            bins = np.hstack((np.arange(indices.shape[0])[:, np.newaxis], gens[indices], dps[indices]))
            family_size_to_counts[family_size][tuple(bins[np.all(bins>=0, axis=1), :].T)] += 1

        if j%10000==0:
            print(j)
        j += 1
    print(i)


# write to file
for family_size in args.family_sizes:
    out_file = '%s/chr.%s.famsize.%d.famgen.counts' % (args.out_dir, args.chrom, family_size)
    print('saving to %s' % out_file)
    np.save(out_file, family_size_to_counts[family_size])

