import sys
import numpy as np

order = int(sys.argv[1])

people = []
with open('data/person_vs_taxa.csv', 'r') as f:
    taxa = next(f).strip().split()
    for line in f:
        people.append(line.strip().split(maxsplit=1)[0])

person_taxa = np.loadtxt('data/person_vs_taxa.csv', delimiter='\t', skiprows=1, usecols=list(range(1, len(taxa)+1)))

taxa = []
with open('data/taxa_vs_variants.csv', 'r') as f:
	variants = next(f).strip().split()
	for line in f:
		taxa.append(line.strip().split(maxsplit=1)[0])

taxa_variant = np.loadtxt('data/taxa_vs_variants.csv', delimiter='\t', skiprows=1, usecols=list(range(1, len(variants)+1)))
m, n = taxa_variant.shape

print('Data loaded', len(taxa), len(variants), taxa_variant.shape)

person_variant = np.zeros((len(people), n**order))
for i in range(len(taxa)):
	print(i)
	var = np.where(taxa_variant[i, :])[0]
	p = var.shape[0]
	indices = np.zeros((var.shape[0]**order,), dtype=int)

	for j in range(order):
		indices += np.tile(np.repeat(var, p**(order-1-j)), p**j) * (n**(order-1-j))

	person_variant[:, indices] += np.tile(person_taxa[:, i, np.newaxis], (1, indices.shape[0]))
