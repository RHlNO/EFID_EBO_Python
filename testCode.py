from genalg import Species, Chromosome
from matplotlib import pyplot as plt
import time

def fitness(inGenes: list):
    score =  sum([-abs(gene-0.5) for gene in inGenes])
    return score

if __name__ == '__main__':
    myspecies = Species(fitness, 2, pop=100, mutate=0.4)

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

