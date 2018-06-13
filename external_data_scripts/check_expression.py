import numpy as np

stages = ['fetal', 'infant', 'child', 'teen', 'adult', '50plus']
exps = [[] for s in stages]
min_exps = [[] for s in stages]
max_exps = [[] for s in stages]
overall_exp = [0 for s in stages]
bp_covered = [0 for s in stages]
with open('peaks.txt', 'r') as f:
	for line in f:
		pieces = line.strip().split('\t')
		chrom = pieces[0]
		start = pieces[1]
		end = pieces[2]

		print(chrom, start, end)

		for i, stage in enumerate(stages):
			try:
				with open('expression/%s.%s.%s.%s.wig' % (stage, chrom, start, end)) as f2:
					exp = 0
					max_exp = 0
					min_exp = None
					for line in f2:
						if line.startswith('#'):
							pass
						else:
							pieces = line.strip().split('\t')
							exp += float(pieces[3])*(int(pieces[2])-int(pieces[1]))
							max_exp = max(max_exp, float(pieces[3]))
							min_exp = float(pieces[3]) if min_exp is None else min(min_exp, float(pieces[3]))
					exps[i].append(exp/(int(end)-int(start)+1))
					max_exps[i].append(max_exp)
					min_exps[i].append(0 if min_exp is None else min_exp)
					overall_exp[i] += exp
					bp_covered[i] += (int(end) - int(start) + 1)
			except:
				pass
			
for stage, exp, max_exp, min_exp, bp, over in zip(stages, exps, max_exps, min_exps, bp_covered, overall_exp):
	print(stage, 'mean', np.mean(exp), 'median', np.median(exp))
	print('max', 'mean', np.mean(max_exp), 'max', np.median(max_exp))
	print('min', 'mean', np.mean(min_exp), 'max', np.median(min_exp))
	print('has_expression', len([x for x in max_exp if x > 0])/len(max_exp))
	print('overall', over/bp)
