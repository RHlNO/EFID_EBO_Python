import time
from typing import Callable
from operator import attrgetter
import random
import math
import numpy as np
from .chromosome import Chromosome


class Species:
    """
    A class used to create and manage a Species instance of a genetic algorithm
    ...

    Attributes
    ----------
    population : list of Chromosome
        List containing each chromosome in the population set with genes according to chromSize and gene values (0-1)
    bestChrom : Chromosome
        The best fitness individual of the current generation

    Methods
    -------
    progress()
        Advances the genetic species one generation and returns best chromosome
    """

    def __init__(self, fitfunc: Callable, chromSize: int, pop=200, cross=0.5, mutate=0.1, elite=0.05):
        """
        Parameters
        ----------
        fitfunc : Callable
            Function that takes a chromosome argument and returns a float fitness value to be maximized by the GA
        chromSize : int
            Numer of genes per individual chromosome in the species
        pop = 200 : int
            Number of individual chromosomes per generation in the species
        cross : float
            (0-1) input of the percent rate at which crossover of genes occurs
        mutate : float
            (0-1) input of the percent rate at which mutation of genes occurs
        elite : float
            (0-1) input of the percent rate at which elitism of chromosomes occurs
        """

        self.fitfunc = fitfunc
        self.crossover_rate = cross
        self.mutation_rate = mutate
        self.elitism_rate = elite

        # Generate initial set of chromosomes
        self.population = []
        for idx in range(pop):
            self.population.append(Chromosome(self.fitfunc, chromSize))

        # Sort initial set by fitness
        self.population.sort(key=attrgetter('fitness'))
        self.bestChrom = self.population[-1]

    def progress(self, verbose=False):
        """
        Parameters
        ----------
        verbose : bool
            Specify whether to print generation status output (time, best fitness, etc)
        """

        t_start = time.perf_counter()

        # Create list for to store new population of chromosomes
        newpop = []

        # Create weighted selection odds list for weighted roulette selection
        parent_odds = []
        fits = [chrom.fitness for chrom in self.population]
        fits = [(ftns-min(fits))/np.ptp(fits) for ftns in fits]  # Normalize fitnesses from 0 to 1
        sumfits = sum(fits)
        for fitness in fits:
            try:
                parent_odds.append(fitness/sumfits + parent_odds[-1])
            except IndexError:
                parent_odds.append(fitness/sumfits)

        # Elitism: Copy percentage of the best individuals to new population
        numElite = math.ceil(self.elitism_rate * len(self.population))
        for idx in range(numElite):
            newpop.append(self.population[-idx])

        # Generate new population through chromosome breeding
        for idx in range(numElite, len(self.population)):

            # Selection: Get two parents for new chromosome
            p1 = self.selectparent(parent_odds)
            p2 = self.selectparent(parent_odds)
            while p1 == p2:  # Loop to ensure no duplicate parents selected
                p2 = self.selectparent(parent_odds)

            # Crossover: Supply child with crossover genes from parents
            child = Chromosome(self.fitfunc, len(p2.genes))  # initialize child
            for idx in range(len(child.genes)):
                rv = random.random()
                if rv < self.crossover_rate:
                    child.genes[idx] = p2.genes[idx]
                else:
                    child.genes[idx] = p1.genes[idx]

            # Mutation: Chance to generate new randomized gene for each gene in chromosome
            for idx in range(len(child.genes)):
                rv = random.random()
                if rv < self.mutation_rate:
                    child.genes[idx] = random.random()

            # Add child to new population
            newpop.append(child)

        # Update population with fitness-sorted new generation population
        self.population = sorted(newpop, key=attrgetter('fitness'))
        self.bestChrom = self.population[-1]

        # Print generation statistics if verbose is True
        t_gen = time.perf_counter() - t_start
        if verbose:
            buf = 'Generation time: %f  |  Best fitness: %f' % (t_gen, self.bestChrom.fitness)
            print(buf)

        return self.bestChrom, self.bestChrom.fitness  # Return best individual and fitness of the best elite

    def selectparent(self, oddslist):
        # Weighted roulette selection - higher fitness individuals have proportionally higher chance to be selected
        rv = random.random()
        # compare random float (0-1) to oddslist and find associated chromosome
        for idx, odds in enumerate(oddslist):
            if rv < odds:
                return self.population[idx]
        raise RuntimeError('Parent selection failed to find matching parent in oddslist.')
