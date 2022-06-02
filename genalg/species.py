import time
from typing import Callable
from operator import attrgetter
import random
import math
import numpy as np
from .chromosome import Chromosome

# TODO: CONVERT TO NUMPY ARRAY AND LOGICAL INDEXING AND CHECK PERFORMANCE IMPROVEMENT
# TODO: ADD GENERATION HISTORY


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

    def __init__(self, chromSize: int,
                 fitfunc: Callable,
                 fitargs: tuple = None,
                 pop=200, cross=0.8, mutate=0.1, elite=0.05):
        """
        Parameters
        ----------
        chromSize : int
            Numer of genes per individual chromosome in the species
        fitfunc : Callable
            Function that takes a chromosome argument and returns a float fitness value to be maximized by the GA
        fitargs : list
            List of additional arguments that will be passed to user defined fitness function
        pop = 200 : int
            Number of individual chromosomes per generation in the species
        cross : float
            (0-1) input of the percent rate at which crossover of genes occurs
        mutate : float
            (0-1) input of the percent rate at which mutation of genes occurs
        elite : float
            (0-1) input of the percent rate at which elitism of chromosomes occurs
        """
        self.chromSize = chromSize
        self.fitfunc = fitfunc
        self.fitargs = fitargs
        self.crossover_rate = cross
        self.mutation_rate = mutate
        self.elitism_rate = elite

        # Generate initial set of chromosomes
        self.population = []
        for idx in range(pop):
            self.population.append(self.newChrom())
            self.population[-1].calcFitness()

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
            try:  # construct cumulative selection probability distribution
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
            parents = [None, None]
            parents[0] = self.selectparent(parent_odds)
            parents[1] = self.selectparent(parent_odds)
            while parents[0] == parents[1]:  # Loop to ensure no duplicate parents selected
                parents[1] = self.selectparent(parent_odds)
            parents.sort(key=attrgetter('fitness'))

            # Crossover: Supply child with crossover genes from parents
            child = self.newChrom()  # initialize child
            for idx in range(len(child.genes)):
                rv = random.random()
                if rv < self.crossover_rate:
                    child.genes[idx] = parents[1].genes[idx]
                else:
                    child.genes[idx] = parents[0].genes[idx]

            # Mutation: Chance to generate new randomized gene for each gene in chromosome
            for idx in range(len(child.genes)):
                rv = random.random()
                if rv < self.mutation_rate:
                    child.genes[idx] = random.random()

            # Add child to new population
            newpop.append(child)

        # Update fitness of population individuals
        t_fitcalc = []
        for chrom in newpop:
            t_prefit = time.perf_counter()
            chrom.calcFitness()
            t_fitcalc.append(time.perf_counter()-t_prefit)

        # Update population with fitness-sorted new generation population
        self.population = sorted(newpop, key=attrgetter('fitness'))

        # Get best chromosome from population
        self.bestChrom = self.population[-1]

        # Print generation statistics if verbose is True
        t_gen = time.perf_counter() - t_start
        if verbose:
            buf = 'Generation time: %f  |  Best fitness: %f  |  Mean fitness calculation time: %f' % (t_gen, self.bestChrom.fitness, np.mean(t_fitcalc))
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

    def newChrom(self):
        # Generate new chromosome matching species parameters
        return Chromosome(self.chromSize, self.fitfunc, self.fitargs)
