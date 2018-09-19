import gzip
import numpy as np
import sys

variant_file = sys.argv[1]

chrom_map = dict([(str(chrom), chrom) for chrom in range(1, 23)])
chrom_map['X'] = 23
chrom_map['Y'] = 24
chrom_map['MT'] = 25

with gzip.open(variant_file, 'rt') as f:
    chrom_coord = []
    for line in f:
        chrom, pos, _, ref, alt = line.strip().split('\t', maxsplit=5)[:5]
        is_biallelic_snp = 1 if len(ref) == 1 and len(alt) == 1 and ref != '.' and alt != '.' else 0
        chrom_coord.append((chrom_map[chrom], int(pos), is_biallelic_snp))
        
chrom_coord = np.asarray(chrom_coord, dtype=int)
np.save(variant_file[:-15] + 'coordinates', chrom_coord)
print(chrom_coord.shape, chrom_coord)