"""
Genetic Algorithm (version 3).

Recombination: None.
Mutation: 80%
Readout: KNN
"""


id_method: str = "Debugging" #T1_GA_OnlyMut_KNN"
id_run: str = "1"

# General imports
import pickle
import multiprocessing as mp
from numpy import mean as npMean, std as npStd, min as npMin, max as npMax
from pathlib import Path
from numpy.random import get_state as npGetstate, set_state as npSetstate
from random import setstate, getstate, random
from statistics import mean as pyMean
from sklearn.neighbors import KNeighborsClassifier as KNN

# DEAP imports
from deap import base, creator, tools

# Custom imports for encoding and evolutionary operations
from lib.encoding import Encoding
from lib.operationsEvo import initEncoding, evaluation, xover_blx, mutation_simple, fitting, get_checkpoint, save_checkpoint
from lib.auxiliary import ExecutionTime, split_data
from lib.LSM import validation_sequence

# Loading the dataset for evaluation
with open(f"../data/FR2_jitter.data", "rb") as file:
    data = pickle.load(file)

# Split the dataset
data_evolution, data_validation = split_data(data["Classes"], data, 0.7, 0.3)

# Parameters for Genetic Algorithm
num_channels: int = data["Channels"]  # Number of input channels for encoding
time_sim: float = data["Tmax"] + 10  # Duration of the simulation
pop_size: int = 20  # Population size for each generation
network_sizes: tuple = (20, 21)  # Size of the networks in the encoding
gen_max: int = 100  # Maximum number of generations
t_size: int = 5  # Tournament size for selection
cr: float = 0.0  # Crossover rate (probability of crossover between individuals)
mr: float = 0.8  # Mutation rate (probability of mutating an individual)
elitism: int = 1  # Number of elite individuals to retain each generation
n_workers: int = mp.cpu_count() - 2  # Number of workers for multiprocessing (leaves 2 CPU cores free)
labels = ["C1", "C2"]

# DEAP framework setup: Create fitness and individual classes
creator.create("Fitness", base.Fitness, weights=(1.0,))  # Fitness function with maximization
creator.create("Individual", Encoding, fitness=creator.Fitness)  # Individual class based on encoding

# Setting up the DEAP toolbox with necessary functions
toolbox = base.Toolbox()
toolbox.register("individual", initEncoding, creator.Individual, network_sizes, num_channels)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Initializes population with individuals
toolbox.register("evaluate", evaluation, inputs=data_evolution, time_sim=time_sim, classifier=KNN)  # Evaluation function for individuals
toolbox.register("cross", xover_blx)  # One-point crossover
toolbox.register("mutate", mutation_simple)  # Mutation function
toolbox.register("select", tools.selTournament, tournsize=t_size)  # Tournament selection
toolbox.register("elitism", tools.selBest, k=elitism)  # Elitism to carry forward the best individual
toolbox.register("replace", tools.selBest, k=pop_size - elitism) # Selects the top offspring to replace the population after elitism.
if n_workers > 1:
    pool = mp.Pool(n_workers)  # Pool for multiprocessing
    toolbox.register("map", pool.map)  # Using the pool for parallel mapping

# Statistics to track during evolution
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", npMean)
stats.register("std", npStd)
stats.register("min", npMin)
stats.register("max", npMax)

if __name__ == "__main__":
    # Check if a checkpoint exists for resuming evolution
    if not Path(f"checkpoint/{id_method}_{id_run}.chck").exists():
        gen: int = 0  # Initialize generation count
        # Population initialization.
        pop: list = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)  # Hall of Fame for the best individual
        logbook = tools.Logbook()  # Logbook to track progress
        run_time: ExecutionTime = ExecutionTime()  # Timer for tracking execution
        # Evaluating initial population
        fitting(toolbox, pop)
        hof.update(pop)
        record = stats.compile(pop)  # Collect statistics for the initial population
        logbook.record(gen=0, t=run_time.report_run_time(), e_t=run_time.report_elapsed(), **record)  # Update logbook
        print(f"Initial population: {record} - Elapsed time: {run_time.report_elapsed(): .4} - Num. Neurons: {pyMean([p.nT for p in pop])}")
        run_time.update()  # Update run time
    else:
        # Load from checkpoint if available
        gen, pop, hof, logbook, py_state, np_state = get_checkpoint("checkpoint", id_method, id_run)
        setstate(py_state)
        npSetstate(np_state)
        run_time = ExecutionTime(logbook.select('t')[-1])

    mean_fit = logbook.select("avg")[-1]
    gen += 1
    # Main evolution loop
    while gen <= gen_max and mean_fit < 1.0:
        print('##############################################################################')
        # Selection process
        offspring = toolbox.select(pop, pop_size)  # Select individuals for the next generation
        # Clone selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Crossover operation
        for i, (child1, child2) in enumerate(zip(offspring[::2], offspring[1::2])):
            if cr >= random():  # Apply crossover with probability cr
                toolbox.cross(child1, child2)
                del child1.fitness.values  # Invalidate fitness after crossover
                del child2.fitness.values
        # Mutation operation
        for mutant in offspring:
            if mr >= random():  # Apply mutation with probability mr
                toolbox.mutate(mutant)
                del mutant.fitness.values  # Invalidate fitness after mutation
        # Evaluate offspring that need new fitness values
        new_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitting(toolbox, new_individuals)
        print(f"Finished new evaluation. Elapsed time: {run_time.report_elapsed(): .4} - Num. Neurons {pyMean([p.nT for p in new_individuals])} - Num. Evaluated {len(new_individuals)}")
        # Elitism and replacement.
        pop = toolbox.elitism(pop) + toolbox.replace(offspring)
        hof.update(pop)  # Update Hall of Fame with the current population
        record = stats.compile(pop)  # Collect statistics for the current population
        logbook.record(gen=gen, t=run_time.report_run_time(), e_t=run_time.report_elapsed(), **record)
        print(f"Generation {gen}: {record}")
        print(f"Total run time: {logbook.select('t')[-1]: .4}")
        run_time.update()  # Update run time
        mean_fit = logbook.select("avg")[-1]
        # Update checkpoint
        save_checkpoint("checkpoint", id_method, id_run, gen, pop, hof, logbook, getstate(), npGetstate())
        gen += 1
    # Las checkpoint update
    save_checkpoint("checkpoint", id_method, id_run, gen, pop, hof, logbook, getstate(), npGetstate(), last=True)
    # Close the multiprocessing pool to free resources, if necessary
    if n_workers > 1:
        pool.close()
        pool.join()

    # Validation
    plot_options = {"convergence": True, "time": True, "liquid": True, "separability": True, "spikes": False, "vms": False,
                    "input": False}
    # npSeed()
    validation_sequence(data_validation, labels, pop, logbook, hof, time_sim, "Best", plot_options=plot_options)

    #I could help refine the code for handling resource reuse and clearing memory after evaluations.

