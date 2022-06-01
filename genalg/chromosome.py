import random


class Chromosome:

    def __init__(self, fitfunc, numGenes):
        self.genes = [random.random() for gene in range(numGenes)]
        self.fitness = fitfunc(self.genes)

    def calcFitness(self, fitfunc):
        self.fitness = fitfunc(self.genes)

    # TODO add parent/child and mutant tracking for visualization or analysis of gene transfer