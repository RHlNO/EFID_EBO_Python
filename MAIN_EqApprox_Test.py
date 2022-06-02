from genalg import Species
from efid import FIS_1_In
from matplotlib import pyplot as plt
import numpy as np
import math


# A basic genetically-trained fuzzy equation approximator
# Utilized here to test and benchmark performance of the GA and of the fuzzy inference system builder
# Attempts to fit a sin wave using fuzzy inferencing with 21 membership functions

# Define fitness function for genetic algorithm
def fitness(inGenes: list, addtl_args):
    # Split provided genes into centers and rules arguments for FIS
    mid_idx = int(len(inGenes) / 2)
    centers = inGenes[:mid_idx]
    rules = inGenes[mid_idx:]

    # Get X inputs and true Y values from additional arguments
    X = addtl_args[0]
    Y_act = addtl_args[1]

    # Create Fuzzy Inference System from centers and rules
    eq_approx_fis = FIS_1_In(centers, rules)

    # Calculate Y approximations by evaluating FIS for input X values
    Y_approx = [eq_approx_fis.evaluate(val) for val in X]

    # sum squared error as fitness score (negative to allow maximization)
    score = -sum([(a-b)**2 for a, b in zip(Y_act, Y_approx)])

    return score


if __name__ == '__main__':

    # Create list of X values and evaluate function at them
    X = np.linspace(-math.pi, math.pi, 101)
    Y_act = [math.sin(val) for val in X]

    # Normalize X inputs and actual Y values to 0-1 range to fit GA ranges
    X_norm = [(val-min(X)) / np.ptp(X) for val in X]
    Y_act_norm = [(val-min(Y_act)) / np.ptp(Y_act) for val in Y_act]

    # Initialize Genetic Algorithm with 21 genes per individual for use in FIS (1 input, 21 mf, 21 rules)
    eqApproxSpec = Species(21, fitness, fitargs=(X_norm, Y_act_norm), cross=0.8, pop=200)

    # Setup live visualization
    plt.ion()
    f1, ax1 = plt.subplots()
    line_act, = ax1.plot(X_norm, Y_act_norm)
    line_approx, = ax1.plot(X_norm, np.zeros(np.shape(X_norm)))

    # Advance genetic algorithm in loop
    fits = []
    numGen = 100  # Number of generations
    for gen in range(numGen):
        # Advance generation in genetic algorithm
        bestChrom, bestFit = eqApproxSpec.progress(verbose=True)
        fits.append(bestFit)

        # Use bestChrom to generate FIS and calculate Y_approx
        mid_idx = int(len(bestChrom.genes) / 2)
        centers = bestChrom.genes[:mid_idx]
        rules = bestChrom.genes[mid_idx:]
        eq_approx_fis = FIS_1_In(centers, rules)
        Y_approx_norm = [eq_approx_fis.evaluate(val) for val in X_norm]

        # Update live-view plot with current best elite approximation
        line_approx.set_data(X_norm, Y_approx_norm)
        f1.canvas.draw()
        f1.canvas.flush_events()

    print(' --- Max Generation Reached ------------------')
    print('Best Fitness: %f' % eqApproxSpec.bestChrom.fitness)

    plt.ioff()

    # Plot generational fitness convergence
    f2, ax2 = plt.subplots()
    ax2.plot(fits)
    plt.xlabel('Generation')
    plt.ylabel('Fitness of the best elite')
    plt.show()
