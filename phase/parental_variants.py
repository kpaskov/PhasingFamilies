import numpy as np
from itertools import product

import sys
import json
from itertools import product
import traceback
from os import listdir, path
import numpy as np

from inheritance_states import InheritanceStates
from input_output import pull_families, pull_gen_data_for_individuals, pull_states_from_file
from losses import LazyLoss
from scipy.sparse import csc_matrix, save_npz

# python phase/phase_chromosome.py 22 data/v34.vcf.ped split_gen_ihart 37 phased_ihart parameter_estimation/params/ihart_multiloss_params.json --detect_deletions --family AU0197 2

import argparse

parser = argparse.ArgumentParser(description='Ancestral variants.')
parser.add_argument('phase_dir', type=str, help='Phase directory.')
parser.add_argument('chrom', type=str, help='Chromosome.')
parser.add_argument('start_pos', type=int, help='Interval to consider.')
parser.add_argument('end_pos', type=int, help='Interval to consider.')

args = parser.parse_args()

with open('%s/info.json' % args.phase_dir, 'r') as f:
    info = json.load(f)

if info['detect_deletions']:
    print('Detecting deletions while phasing ...')

if info['detect_consanguinity']:
    print('Detecting parental consanguinity while phasing ...')

with open(info['param_file'], 'r') as f:
    params = json.load(f)

with open('%s/info.json' % info['data_dir'], 'r') as f:
    assembly = json.load(f)['assembly']

with open('data/chrom_lengths%s.json' % assembly, 'r') as f:
    chrom_length = json.load(f)[args.chrom]

# --------------- pull families of interest ---------------
families = pull_families(info['ped_file'])

# make sure at least one individual has genetic data (chromosome 1 chosen arbitrarily)
sample_file = '%s/samples.json' % info['data_dir']
with open(sample_file, 'r') as f:
    sample_ids = set(json.load(f))

for family in families:
    to_be_removed = [x for x in family.individuals if x not in sample_ids or (x not in params and '%s.%s' % (family.id, x) not in params)]
    family.prune(to_be_removed)

families = [x for x in families if x.num_descendents()>0]
print(len(families), 'have genomic data and parameters')

# we can only work with nuclear families for this analysis
families = [x for x in families if x.num_ancestors()==2 and len(x.ordered_couples)==1]

if info['detect_consanguinity']:
    
    # to detect consanguinity, model parents as siblings - they have the freedom to inherit
    # completely different copies from "mat_shared_ancestor" and "pat_shared_ancestor"
    # or they can have consanguineous regions.
    for family in families:
        family.add_child(family.mat_ancestors[0], 'mat_shared_ancestor', 'pat_shared_ancestor')
        family.add_child(family.pat_ancestors[0], 'mat_shared_ancestor', 'pat_shared_ancestor')

# limit to families with phasing data
families = [family for family in families if path.isfile('%s/%s.phased.txt' % (args.phase_dir, family.id))]

print('Families of interest', len(families))

with open('%s/chr.%s.%d.%d.families.json' % (args.phase_dir, args.chrom, args.start_pos, args.end_pos), 'w+') as f:
    json.dump([x.id for x in families], f)

af_boundaries = np.arange(-np.log10(0.25), info['max_af_cost'], np.log10(2)).tolist()
af_boundaries.extend([-np.log10(1-(10.0**-x)) for x in af_boundaries[1:]])
af_boundaries = np.array(sorted(af_boundaries, reverse=True))
print('af boundaries', af_boundaries)

# estimate ancestral variants for each family
i_s, j_s, v_s, d_s = [], [], [], []
for fam_index, family in enumerate(families):
    try:
        print('family', family.id)

        # create inheritance states
        inheritance_states = InheritanceStates(family, info['detect_deletions'], info['detect_deletions'], info['num_loss_regions'])

        # create loss function for this family
        loss = LazyLoss(inheritance_states, family, params, info['num_loss_regions'], af_boundaries)
        #print('loss created')

        # pull genotype data for this family
        family_genotypes, family_snp_positions, mult_factor = pull_gen_data_for_individuals(info['data_dir'], af_boundaries, assembly, args.chrom, family.individuals, start_pos=args.start_pos, end_pos=args.end_pos)

        # pull phase information
        with open('%s/%s.phased.txt' % (args.phase_dir, family), 'r') as phasef:
            final_states = pull_states_from_file(phasef, args.chrom, family_snp_positions)


        # limit to sites with variants - move from intervals to individual positions
        has_variant = np.any(family_genotypes[:-1, :]>0, axis=0)

        family_genotypes = np.repeat(family_genotypes[:, has_variant], mult_factor[has_variant], axis=1)
        final_states = np.repeat(final_states[:, has_variant], mult_factor[has_variant], axis=1)
        family_snp_positions = np.hstack([np.arange(x[0], x[1]) for x in family_snp_positions[has_variant, :]])
        assert family_genotypes.shape[1] == family_snp_positions.shape[0]
        assert final_states.shape[1] == family_snp_positions.shape[0]

        # update loss cache
        loss.set_cache(family_genotypes)

        # pull ancestral alleles
        ancestral_alleles = np.array([inheritance_states.get_perfect_matches(state)[1] for state in inheritance_states])

        # now figure out probability of alleles
        ancestral_variant_probs = np.zeros((ancestral_alleles.shape[2], 3, family_genotypes.shape[1]))
        for i, (gen, state) in enumerate(zip(family_genotypes.T, final_states.T)):
            missing_indices = [i for i, x in enumerate(state) if x==-1]
            possible_states = np.tile(state, (pow(2, len(missing_indices)), 1))
            possible_states[:, missing_indices] = list(product(*[[0, 1] if (i<4) or (i%2==0) else [2, 3] for i in missing_indices]))
            possible_state_indices = np.array([inheritance_states.index_from_full_state_tuple(state) for state in possible_states])
                
            for j, state_index in enumerate(possible_state_indices):
                ps = loss.ancestral_variant_probabilities(gen, state_index)
                als = ancestral_alleles[state_index, :, :]

                ancestral_variant_probs[np.tile(np.arange(ancestral_alleles.shape[2]), (als.shape[0], 1)).flatten(),
                                            als.flatten(),
                                            i*np.ones((als.shape[0]*als.shape[1],), dtype=int)] += np.tile(ps, (als.shape[1], 1)).T.flatten()
            p_b = np.sum(ancestral_variant_probs[:, :, i], axis=1)
            for k in range(np.max(ancestral_alleles)+1):
                ancestral_variant_probs[p_b!=0, k, i] = ancestral_variant_probs[p_b!=0, k, i]/p_b[p_b!=0]

        for anc_index in range(ancestral_variant_probs.shape[0]):
            i_s.append((fam_index*4 + anc_index)*np.ones(family_genotypes.shape[1],))
            j_s.append(family_snp_positions)
            v_s.append(ancestral_variant_probs[anc_index, 1, :])
            d_s.append(ancestral_variant_probs[anc_index, 2, :])

        print('Done!')
    except Exception: 
        traceback.print_exc()

# save to file
save_npz('%s/chr.%s.%d.%d.parental_variants' % (args.phase_dir, args.chrom, args.start_pos, args.end_pos), 
    csc_matrix((np.hstack(v_s), (np.hstack(i_s), np.hstack(j_s))), shape=(4*len(families), chrom_length)))
save_npz('%s/chr.%s.%d.%d.parental_deletions' % (args.phase_dir, args.chrom, args.start_pos, args.end_pos), 
    csc_matrix((np.hstack(d_s), (np.hstack(i_s), np.hstack(j_s))), shape=(4*len(families), chrom_length)))
    
