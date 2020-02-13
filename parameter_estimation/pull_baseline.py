import compare_rates
import numpy as np
import sys

data_dir = sys.argv[1]

chroms = [str(i) for i in range(1, 23)]
gens = ['0/0', '0/1', '1/1']
obss = ['0/0', '0/1', '1/1', './.']
samples = compare_rates.pull_samples(data_dir, chroms)
baseline_counts = compare_rates.pull_baseline_counts(samples, data_dir, chroms, gens, obss)
np.save('%s/baseline_counts.npy' % data_dir, baseline_counts)