from collections import defaultdict, namedtuple
import sys
import json
import numpy as np
import math

phase_dir = sys.argv[1] #'../phased_ihart'
family_sizes = [int(x) for x in sys.argv[2].split(',')] #'3,4,5,6'
identicals_file = sys.argv[3] #'../sibpair_similarity/identicals.txt'
chrom = sys.argv[4]
outdir = sys.argv[5] # ../deletions_ihart

is_asym = (len(sys.argv) > 6) and sys.argv[6] == '--asym'
print('is asymetric', is_asym)

deletion_share_cutoff=0.9
print('deletion share cutoff', deletion_share_cutoff)

# pull twins
identicals = set()
with open(identicals_file, 'r') as f:
	for line in f:
		pieces = line.strip().split('\t')
		identicals.update(pieces)
print('identicals', len(identicals))

# pull families and figure out how many non-identical kids they have
family_to_individuals = dict()
families_left_out = set()

for j in family_sizes:
	with open('%s/chr.%s.familysize.%d.families.txt' % (phase_dir, 'X' if chrom=='PAR1' else chrom, j), 'r') as f:
		next(f) # skip header
		for line in f:
			pieces = line.strip().split('\t')
			family_key = pieces[0]
			individuals = pieces[1:(1+j)]

			if np.all([x not in identicals for x in individuals[2:]]):
				family_to_individuals[family_key] = individuals
			else:
				families_left_out.add(family_key)
print('families', len(family_to_individuals))
print('families removed due to identicals', len(families_left_out))

individuals = sorted(sum(family_to_individuals.values(), []))
children = set(sum([v[2:] for v in family_to_individuals.values()], []))
ind_to_index = dict([(x, i) for i, x in enumerate(individuals)])
print('individuals', len(individuals))

with open('%s/individuals.txt' % outdir, 'w+') as f:
	for ind in individuals:
		f.write('%s\t%s\n' % (ind, 'child' if ind in children else 'parent'))


# pull phase data for each family
family_to_states = defaultdict(list)
family_to_pos = defaultdict(list)
family_to_indices = defaultdict(list)
for j in family_sizes:
	with open('%s/chr.%s.familysize.%d.phased.txt' % (phase_dir, 'X' if chrom=='PAR1' else chrom, j), 'r')  as f:
		next(f) # skip header

		for line in f:
			pieces = line.strip().split('\t')
			family_key = pieces[0]
			state = [int(x) for x in pieces[1:(2+2*j+2*(j-2))]]
			start_pos, end_pos = [int(x) for x in pieces[(2+2*j+2*(j-2)):(4+2*j+2*(j-2))]]
			start_index, end_index = [int(x) for x in pieces[(4+2*j+2*(j-2)):(6+2*j+2*(j-2))]]
			assert end_pos >= start_pos

			family_to_states[family_key].append(state)
			family_to_pos[family_key].append((start_pos, end_pos))
			family_to_indices[family_key].append((start_index, end_index))
print('pulled phase data')

# read in deletions
Deletion = namedtuple('Deletion', ['family', 'chrom', 'start_pos', 'end_pos', 'length', 'phase_length',
	'opt_start_pos', 'opt_end_pos', 
	'trans', 'notrans', 'family_size', 'is_mat', 'is_pat', 'is_denovo', 'mother', 'father'])

deletions = []

for family_key, states in family_to_states.items():
	if family_key in family_to_individuals:
		states = np.asarray(states)
		assert np.all(states[0, :4] != 0)
		assert np.all(states[-1, :4] != 0)
		positions = np.asarray(family_to_pos[family_key])
		phase_indices = np.asarray(family_to_indices[family_key])
		inds = family_to_individuals[family_key]
		j = len(inds)

		# ------------------ Inherited Deletions ---------------------
		# for each ancestral chromosome
		if chrom != 'X':
			anc_chroms = range(4)
		else:
			anc_chroms = range(2)
			
		for anc in anc_chroms:
			is_mat = anc==0 or anc==1
			is_pat = anc==2 or anc==3

			start_indices = np.where((states[:-1, anc] != 0) & (states[1:, anc] == 0))[0]+1
			end_indices = np.where((states[:-1, anc] == 0) & (states[1:, anc] != 0))[0]+1
			for s_ind, e_ind in zip(start_indices, end_indices):

				# check if recombination event occured and that inheritance state is known
				has_recomb = False
				if is_mat:
					indices = np.arange(4, states.shape[1]-1, 2)
				else:
					indices = np.arange(5, states.shape[1]-1, 2)

				inh_known = np.all(states[s_ind:e_ind, indices] != -1)

				for i in range(s_ind, e_ind):
					if np.any(states[i, indices] != states[s_ind, indices]):
						has_recomb = True

				# check if in ok region
				ok_region = np.all(states[s_ind:e_ind, -1] == 0)

				start_pos, end_pos = positions[s_ind, 0], positions[e_ind-1, 1]
				length = int(end_pos - start_pos + 1)
				if ok_region and inh_known and (not has_recomb) and (chrom != 'PAR1' or start_pos <= 2699520):
					start_pos, end_pos = positions[s_ind, 0], positions[e_ind-1, 1]
					start_phase_index, end_phase_index = phase_indices[s_ind, 0], phase_indices[e_ind-1, 1]

					# find boundaries of the deletion
					opt_start_index = s_ind
					while states[opt_start_index, anc] != 1 and opt_start_index > 0:
						opt_start_index -= 1
					opt_start_pos = positions[opt_start_index+1, 0]

					opt_end_index = e_ind
					while(states[opt_end_index, anc]) != 1 and opt_end_index < states.shape[0]-1:
						opt_end_index += 1
					opt_end_pos = positions[opt_end_index-1, 1]

					assert start_pos <= end_pos
					assert opt_start_pos <= start_pos
					assert end_pos <= opt_end_pos

					state = states[s_ind, :]

					# children
					trans, notrans = [], []
					for k, child in zip(range(2, j), inds[2:]):
						mom_s, dad_s = state[(2*k):(2*(k+1))]

						if is_mat:
							assert mom_s != -1
							if anc==mom_s:
								trans.append(child)
							else:
								notrans.append(child)
						if is_pat:
							assert dad_s != -1
							if anc==2+dad_s:
								trans.append(child)
							else:
								notrans.append(child)

					if len(trans) + len(notrans) == j-2:
						deletions.append(Deletion(family_key, chrom, int(start_pos), int(end_pos), length,
							int(end_phase_index-start_phase_index+1),
							int(opt_start_pos), int(opt_end_pos), tuple(trans), tuple(notrans),
							len(inds), is_mat, is_pat, False,
							inds[0], inds[1]))

		# ------------------ De novo Deletions ---------------------
		if chrom != 'X':
			anc_chroms = range(2*j, 2*j + 2*(j-2))
		else:
			anc_chroms = range(2*j, 2*j + 2*(j-2), 2)
		# for each child
		for anc in anc_chroms:
			is_mat = anc%2 == 0
			is_pat = anc%2 == 1

			start_indices = np.where((states[:-1, anc] != 1) & (states[1:, anc] == 1))[0]+1
			end_indices = np.where((states[:-1, anc] == 1) & (states[1:, anc] != 1))[0]+1
			for s_ind, e_ind in zip(start_indices, end_indices):

				# check if recombination event occured and that inheritance state is known
				has_recomb = False
				if is_mat:
					indices = np.arange(4, states.shape[1]-1, 2)
				else:
					indices = np.arange(5, states.shape[1]-1, 2)

				inh_known = np.all(states[s_ind:e_ind, indices] != -1)

				for i in range(s_ind, e_ind):
					if np.any(states[i, indices] != states[s_ind, indices]):
						has_recomb = True

				# check if in ok region
				ok_region = np.all(states[s_ind:e_ind, -1] == 0)

				start_pos, end_pos = positions[s_ind, 0], positions[e_ind-1, 1]
				length = int(end_pos - start_pos + 1)
				if ok_region and inh_known and (not has_recomb):
					start_pos, end_pos = positions[s_ind, 0], positions[e_ind-1, 1]
					start_phase_index, end_phase_index = phase_indices[s_ind, 0], phase_indices[e_ind-1, 1]

					# find boundaries of the deletion
					opt_start_index = s_ind
					while states[opt_start_index, anc] != 0 and opt_start_index > 0:
						opt_start_index -= 1
					opt_start_pos = positions[opt_start_index+1, 0]

					opt_end_index = e_ind
					while(states[opt_end_index, anc]) != 0 and opt_end_index < states.shape[0]-1:
						opt_end_index += 1
					opt_end_pos = positions[opt_end_index-1, 1]

					assert start_pos <= end_pos
					assert opt_start_pos <= start_pos
					assert end_pos <= opt_end_pos

					state = states[s_ind, :]

					# children
					trans, notrans = [inds[2+int(math.floor((anc - 2*j)/2))]], []

					deletions.append(Deletion(family_key, chrom, int(start_pos), int(end_pos), length,
							int(end_phase_index-start_phase_index+1),
							int(opt_start_pos), int(opt_end_pos), tuple(trans), tuple(notrans),
							len(inds), is_mat, is_pat, True,
							inds[0], inds[1]))

deletions = sorted(deletions, key=lambda x: x.start_pos)
print('deletions', len(deletions))

# write to json
json_deletions = list()
for deletion in deletions:
	json_deletions.append(deletion._asdict())

with open('%s/chr.%s.deletions.json' % (outdir, chrom), 'w+') as f:
	json.dump(json_deletions, f, indent=4)

# create collections
class DeletionCollection:
	def __init__(self, deletion, matches):
		self.deletion = deletion
		self.matches = matches

collections = []
    
starts = np.array([d.start_pos for d in deletions])
stops = np.array([d.end_pos for d in deletions])

ordered_start_indices = np.argsort(starts)
ordered_starts = starts[ordered_start_indices]
ordered_stop_indices = np.argsort(stops)
ordered_stops = stops[ordered_stop_indices]
        
insert_starts_in_stops = np.searchsorted(ordered_stops, starts)
insert_stops_in_starts = np.searchsorted(ordered_starts, stops, side='right')
        
indices = np.ones((len(deletions),), dtype=bool)

for del_index, main_d in enumerate(deletions):
	indices[:] = True
	indices[ordered_stop_indices[:insert_starts_in_stops[del_index]]] = False
	indices[ordered_start_indices[insert_stops_in_starts[del_index]:]] = False

	collections.append(DeletionCollection(main_d, [deletions[j] for j in np.where(indices)[0]]))
print('collections', len(collections))

# prune deletions
for c in collections:
	lengths = np.array([d.length for d in c.matches])
	overlaps = np.array([min(d.end_pos, c.deletion.end_pos)-max(d.start_pos, c.deletion.start_pos)+1 for d in c.matches])
	if is_asym:
		c.matches = set([c.matches[j] for j in np.where((overlaps >= deletion_share_cutoff*c.deletion.length))[0]])
	else:
		c.matches = set([c.matches[j] for j in np.where((overlaps >= deletion_share_cutoff*lengths) & (overlaps >= deletion_share_cutoff*c.deletion.length))[0]])

print('deletions pruned')

# prune collections (get rid of collections that are identical to other collections)
deletion_to_index = dict([(x.deletion, i) for i, x in enumerate(collections)])
    
for c in collections:
	if c is not None:
		for d in c.matches:
			index = deletion_to_index[d]
			if (c.deletion != d) and (collections[index] is not None) and (c.matches == collections[index].matches):
				collections[index] = None
print('collections pruned, removed %d of %d' % (len([x for x in collections if x is None]), len(collections)))
collections = [x for x in collections if x is not None]

# write to json
json_collections = list()
for collection in collections:
	json_collections.append({
		'deletion': collection.deletion._asdict(),
		'matches': [m._asdict() for m in collection.matches]
	})
with open('%s/chr.%s.collections.json' % (outdir, chrom), 'w+') as f:
	json.dump(json_collections, f, indent=4)
    



