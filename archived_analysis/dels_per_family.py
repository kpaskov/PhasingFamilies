import math

phase_dir = 'sherlock_phased'

d_length = [0, 0, 0, 0]
d_intervals = []
families = set()
for chrom in range(1, 23):
    print(chrom)
    for j in [3, 4, 5, 6, 7]:
        # load deletions
        with open('%s/chr.%d.familysize.%d.phased.txt' % (phase_dir, chrom, j), 'r')  as f:
            next(f) # skip header
            for line in f:
                pieces = line.strip().split('\t')
                families.add(pieces[0])
                inheritance_state = [None if x == '*' else int(x) for x in pieces[1:(1+(j*2))]]
                start_pos, end_pos, start_index, end_index = [int(x) for x in pieces[(1+(j*2)):(5+(j*2))]]
                length = end_pos - start_pos + 1

                for i in range(4):
                    if inheritance_state[i] == 1:
                        d_length[i] += length
                    elif d_length[i] != 0:
                        if d_length[i] > 100:
                            d_intervals.append(d_length[i])
                        d_length[i] = 0
        for i in range(4):
            if d_length[i] > 100:
                d_intervals.append(d_length[i])
    

# length of deletions
max_length = max(d_intervals)
min_length = min(d_intervals) 
print('min length', min_length, 'max length', max_length, 'num', len(d_intervals), 'num/family', len(d_intervals)/len(families))
print(sorted(d_intervals, reverse=True)[:200])