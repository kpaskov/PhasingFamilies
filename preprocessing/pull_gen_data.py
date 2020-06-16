from collections import defaultdict
import numpy as np
from scipy.sparse import csc_matrix, save_npz
import time
import gzip
from itertools import product
import sys
import argparse

parser = argparse.ArgumentParser(description='Pull genotypes.')
parser.add_argument('vcf_file', type=str, help='VCF file to pull from.')
parser.add_argument('out_directory', type=str, help='Output directory.')
parser.add_argument('chrom', type=str, help='Chromosome of interest.')
parser.add_argument('--batch_size', type=int, default=None, help='Restrict number of positions per file to batch_size.')
parser.add_argument('--batch_num', type=int, default=0, help='To be used along with batch_size to restrict number of positions per file. Will use positions[(batch_num*batch_size):((batch_num+1)*batch_size)]')
parser.add_argument('--maxsize', type=int, default=500000000, help='Amount of memory per block.')
args = parser.parse_args()

t0 = time.time()

chrom_int = 23 if args.chrom == 'X' else 24 if args.chrom == 'Y' else 25 if args.chrom == 'MT' else int(args.chrom)

# Pull data from vcf
def process_vcf(f):
    with gzip.open('%s/chr.%s.%d.gen.variants.txt.gz' % (args.out_directory, args.chrom, args.batch_num), 'wt') as variant_f:
        # Skip header
        line = next(f)
        while line.startswith('##'):
            line = next(f)

        with open('%s/chr.%s.gen.samples.txt' % (args.out_directory, args.chrom), 'w+') as sample_f:
            # Pull sample_ids and write to file
            sample_ids = line.strip().split('\t')[9:]
            sample_f.write('\n'.join(sample_ids))
            print('Num individuals with genomic data', len(sample_ids))

        # Pull genotypes from vcf
        m = len(sample_ids)
        data, indices, indptr, index = np.zeros((args.maxsize,), dtype=np.int8), np.zeros((args.maxsize,), dtype=int), [0], 0
        gen_mapping = {'./.': -1, '0/0': 0, '0|0': 0, '0/1': 1, '0|1': 1, '1/0': 1, '1|0': 1, '1/1': 2, '1|1': 2}

        # enumerate all chrom options
        chrom_options = [args.chrom, 'chr'+args.chrom]
        if args.chrom == 'X':
            chrom_options = chrom_options + ['23', 'chr23']
        if args.chrom == 'Y':
            chrom_options = chrom_options + ['24', 'chr24']
        if args.chrom == 'MT':
            chrom_options = chrom_options + ['25', 'chr25']
        
        line = next(f)
        num_lines_in_file = 0
        current_line_index = 0
        chrom_coord = []
        for j, line in enumerate(f):
            pieces = line.split('\t', maxsplit=1)

            if pieces[0] in chrom_options:

                if (current_line_index >= args.batch_num*args.batch_size) and (current_line_index < (args.batch_num+1)*args.batch_size):
                    pieces = line.strip().split('\t')
                    format = pieces[8].strip().split(':')

                    # Write variant to file
                    variant_f.write('\t'.join(pieces[:9]) + '\n')

                    pos, _, ref, alt = pieces[1:5]
                    is_biallelic_snp = 1 if len(ref) == 1 and len(alt) == 1 and ref != '.' and alt != '.' else 0
                    is_pass = pieces[6] == 'PASS'
                    chrom_coord.append((chrom_int, int(pos), is_biallelic_snp, is_pass))

                    # Pull out genotypes
                    gen_index = format.index('GT')
                    for i, piece in enumerate(pieces[9:]):
                        segment = piece.split(':', maxsplit=gen_index+1)
                        gt = gen_mapping.get(segment[gen_index], -1) # For now we mark multi-base loci as unknown

                        if gt != 0:
                            indices[index] = i
                            data[index] = gt
                            index += 1
                    indptr.append(index)
                    num_lines_in_file += 1

                    # If file has gotten really big, raise error.
                    if index+m >= args.maxsize:
                        raise Exception('Maxsize too low. Increase maxsize and rerun.')

                current_line_index += 1

    if index>0:
        gen = csc_matrix((data[:index], indices[:index], indptr), shape=(m, num_lines_in_file), dtype=np.int8)
            
        # Save to file
        save_npz('%s/chr.%s.%d.gen' % (args.out_directory, args.chrom, args.batch_num), gen)
        np.save('%s/chr.%s.%d.gen.coordinates' % (args.out_directory, args.chrom, args.batch_num), np.asarray(chrom_coord, dtype=int))
    else:
        print('No data in batch.')
    print('Completed in ', time.time()-t0, 'sec')

if args.vcf_file.endswith('.gz'):
    with gzip.open(args.vcf_file, 'rt') as f:
        process_vcf(f)
else:
    with open(args.vcf_file, 'r') as f:
        process_vcf(f)

