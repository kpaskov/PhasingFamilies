import numpy as np
import scipy.stats

ps = np.arange(0, 1.001, 0.001)

# children = 1, 2, 3, 4, 5
qs = np.zeros((ps.shape[0], 5))
qs[0, :] = [1/(x+1) for x in range(5)]
for i in range(5):
    qs[1:, i] = ps[1:]/(1-(1-ps[1:])**(i+1))

prior = scipy.stats.beta.pdf(ps, 2, 2)
    
def calc_posterior(trans, notrans, num_sizes=5):
    Ls = np.ones((ps.shape[0], num_sizes))    
    for i in range(1, num_sizes):
        #Ls[:, i] = np.power(qs[:, i], trans[i]) * np.power(1-qs[:, i], notrans[i])
        #Ls[:, i] = Ls[:, i]/np.sum(Ls[:, i])
        Ls[0:-1, i] = trans[i] * np.log(qs[0:-1, i]) + notrans[i]*np.log(1-qs[0:-1, i])
        if notrans[i] == 0:
            Ls[-1, i] = trans[i] * np.log(qs[-1, i])
        else:
            Ls[-1, i] = -np.inf

    likelihood = np.sum(Ls, axis=1)
    likelihood = likelihood - np.nanmax(likelihood)
    likelihood = np.exp(likelihood)

    norm = np.sum(likelihood)
    assert norm > 0 
    likelihood = likelihood/norm

    posterior = likelihood * prior
    return posterior*ps.shape[0]/np.sum(posterior)

def calc_trans_rate(trans, notrans, na_cutoff=5, num_sizes=5):    
    if np.sum(trans) + np.sum(notrans) < na_cutoff:
        return np.nan
    
    posterior = calc_posterior(trans, notrans, num_sizes=num_sizes)
    return np.mean(ps*posterior)

def calculate_transmission_rates(contingency, family_sizes, na_cutoff=5):
    # chrom, collection, family_size
    trans_rates = np.ones((contingency.shape[0],))
    for j in range(contingency.shape[0]):
        trans_rates[j] = calc_trans_rate(contingency[j, :, 1], contingency[j, :, 0], na_cutoff=na_cutoff, num_sizes=len(family_sizes))
    return trans_rates

def calculate_posteriors(contingency, family_sizes, verbose=True):
    # chrom, collection, family_size
    posteriors = np.ones((contingency.shape[0], ps.shape[0]))
    for j in range(contingency.shape[0]):
        posteriors[j, :] = calc_posterior(contingency[j, :, 1], contingency[j, :, 0], num_sizes=len(family_sizes))
    return posteriors

def calculate_posterior_overlap(aff_posterior, unaff_posterior):
    # chrom, collection, family_size
    overlaps = np.ones((aff_posterior.shape[0],))

    for j in range(aff_posterior.shape[0]):
        overlaps[j] = np.sum(np.minimum(aff_posterior[j, :], unaff_posterior[j, :]))/np.sum(aff_posterior[j, :])
    return overlaps

def calculate_posterior_pvalue(posterior):
    # chrom, collection, family_size
    posterior_pvalues = np.ones((posterior.shape[0], 2))
    for j in range(posterior.shape[0]):
        posterior_pvalues[j, 0] = np.sum(posterior[j, ps<=0.5])/np.sum(posterior[j, :])
        posterior_pvalues[j, 1] = np.sum(posterior[j, ps>=0.5])/np.sum(posterior[j, :])
    return posterior_pvalues

    