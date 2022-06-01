from genalg import Species
from matplotlib import pyplot as plt
import numpy as np

def fitness(inGenes: list, addtl_args):
    score =  np.mean([-(gene-0.5)**2 for gene in inGenes])
    return score

if __name__ == '__main__':
    myspecies = Species(10, fitness, pop=200)

    fits = []

    numGen = 1000  #Number of generations
    for gen in range(numGen):
        bestChrom, bestFit = myspecies.progress(verbose=True)
        fits.append(bestFit)

    print(myspecies.bestChrom.fitness)
    print(myspecies.bestChrom.genes)

    plt.plot(fits)
    plt.xlabel('Generation')
    plt.ylabel('Fitness of the best elite')
    plt.show()

