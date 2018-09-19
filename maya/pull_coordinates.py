import gzip
import numpy as np
import sys

variant_file = sys.argv[1]

chrom_map = {'1': 1, '2': 2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10,
'11':11, '12':12, '13':13, '14':14, '15':15, '16':16, '17':17, '18':18, '19':19, '20':20,
'21':21, '22':22, 'X':23, 'Y':24, 'MT':25}

with gzip.open(variant_file, 'rt') as f:
    chrom_coord = []
    for line in f:
        chrom, pos, _, ref, alt = line.strip().split('\t', maxsplit=5)[:5]
        is_biallelic_snp = 1 if len(ref) == 1 and len(alt) == 1 and ref != '.' and alt != '.' else 0
        chrom_coord.append((chrom_map[chrom], int(pos), is_biallelic_snp))
        
chrom_coord = np.asarray(chrom_coord, dtype=int)
np.save(variant_file[:-15] + 'coordinates', chrom_coord)
print(chrom_coord.shape, chrom_coord)