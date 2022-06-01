import random


class Chromosome:

    def __init__(self, numGenes, fitfunc, fitargs):
        self.genes = [random.random() for gene in range(numGenes)]
        self.fitfunc = fitfunc
        self.fitargs = fitargs

    def calcFitness(self):
        self.fitness = self.fitfunc(self.genes, self.fitargs)

    # TODO add parent/child and mutant tracking for visualization or analysis of gene transfer