"""
Neural Architecture Search (NAS) based in Genetic Algorithm (GA) for Liquid State Machines (LSM).
This approach is intended for testing FR5, PR4, PR8, PR12 to be published in the Special Issue - MCA2025.

Variables:
    ds: Name of the Dataset = Depending on the dataset.
    id_method: Id of the method. Intended to identify results and checkpoints.
    id_run: ID of the run. Intended to identify the run of the method.
    n_workers: Number of parallel process = cpu count (cpu_count - 2 for local experiments) # 1 for non-parallel experiments.
    freq_check: Frequency for updating the checkpoints = 5.
    labels: List of name of labels = Depending on the dataset. Intended for visualization purposes.
    pop_size: Population size = 20
    gen_max: Max number of generations = 100
    t_size: Tournament size = 5
    cr: Crossover rate = 0.5 # Blend Crossover (BLX)
    mr: Mutation rate = 0.8 #
    elitism: Number of individuals to be preserved = 1
    channels: Number of input channels =  Depending on the dataset.
    sim_time: Duration of NEST simulation = Depending on the dataset.
    net_size: Number of neurons in the liquid = (20, 21) # Mostly 20.
"""
# General imports
import pickle
from multiprocessing import cpu_count, Pool
from sklearn.linear_model import Perceptron
from numpy import mean as npMean, std as npStd, min as npMin, max as npMax
from pathlib import Path
from statistics import mean as pyMean, stdev as pyStdev
from random import setstate as pySetstate, random as pyRandom, getstate as pyGetstate
from numpy.random import set_state as npSetstate, get_state as npGetstate
import psutil
# DEAP imports
from deap import base, creator, tools
# Source imports
from lib.encoding import Encoding
from lib.operationsEvo import initEncoding, evaluation, xover_blx, mutation_simple, fitting, get_checkpoint, save_checkpoint
from lib.auxiliary import ExecutionTime, split_data
from lib.LSM import validation_sequence

# Create fitness and individual classes.
creator.create("Fitness", base.Fitness, weights=(1.0,))  # Fitness maximization.
creator.create("Individual", Encoding, fitness=creator.Fitness) # Individual class based on encoding.

def main(ds_name: str, id_method: str, id_run: str, n_workers: int) -> tuple[float, tuple, base.Toolbox]:
    """
    Main algorithm.
    :param ds_name: Name of the dataset.
    :param id_method: Name of the method.
    :param id_run: ID of the run of the method.
    :param n_workers: Number of parallel process.
    :return: None
    """
    # ---- General Parameters:
    method: str = f"{id_method}-{ds_name}" # ID of the method. Intended to identify results and checkpoints.
    freq_check: int = 5 # Frequency of saving checkpoints.
    # Load Dataset
    with open(f"../data/{ds_name}.data", "rb") as file:
        data = pickle.load(file)
    # Separation of data:
    # Train-Test = 70% | Validation = 30% (p_validation)
    # Train = 70% (p_train) | Test = 30%
    data_evolve, data_val = split_data(classes=data["Classes"], data=data, p_validation=0.3, p_train=0.7)
    labels: list = data["Labels Names"] # List of name of labels.
    # ---- GA Parameters:
    pop_size: int = 50 # Population size.
    gen_max: int = 100 # Max number of generations.
    t_size: int = int(0.25 * pop_size) # Tournament size.
    cr: float = 0.8 # Crossover rate.
    mr: float = 0.5 # Mutation rate.
    elitism: int = int(0.05 * pop_size) # Number of individuals to be preserved.
    # ---- LSM Parameters:
    channels: int = data["Channels"] # Number of input channels.
    sim_time: int = data["Tmax"] + 10 # Duration of NEST simulation.
    net_size: tuple | int = 20 # Number of neurons in the liquid.
    # ---- DEAP Framework:
    # DEAP toolbox.
    toolbox = base.Toolbox()
    toolbox.register("individual", initEncoding, encoding=creator.Individual, dimensions=net_size, channels=channels) # Set individual.
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Set population with individuals.
    toolbox.register("evaluate", evaluation, inputs=data_evolve, time_sim=sim_time, classifier=Perceptron) # Evaluation function for individuals.
    toolbox.register("crossover", xover_blx) # BLX.
    toolbox.register("mutate", mutation_simple) # Mutation (no increase on liquid size).
    toolbox.register("select", tools.selTournament, tournsize=t_size) # Tournament selection.
    toolbox.register("elitism", tools.selBest, k=elitism) # Perform elitism.
    toolbox.register("replace", tools.selBest, k=pop_size - elitism) # Replacement of population
    if n_workers > 1:
        pool = Pool(n_workers) # Create pool for multiprocessing.
        toolbox.register("map", pool.map)  # Replace the map function for the multiprocessing version.
    # ---- Statistics setup:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", npMean)
    stats.register("std", npStd)
    stats.register("min", npMin)
    stats.register("max", npMax)
    # Check if a checkpoints exists to resume the evolution.
    if not Path(f"checkpoints/{method}_{id_run}.chck").exists():
        gen: int = 0 # Initialize generation count.
        pop: list = toolbox.population(n=pop_size) # Population initialization.
        hof = tools.HallOfFame(1) # Hall of Fame for the best individual.
        logbook = tools.Logbook() # Initialize Logbook to track progress.
        time: ExecutionTime = ExecutionTime() # Timer for tracking execution.
        # Evaluate Initial population.
        fitting(toolbox, pop)
        hof.update(pop) # Update HOF.
        record_stats = stats.compile(pop) # Collect statistics for initial population.
        logbook.record(gen=gen, t=time.report_run_time(), elapse_time=time.report_elapsed(),
                       evals=pop_size, **record_stats) # Update logbook.
        print(f"Initial Population (Gen 0): {record_stats} - Elapsed Time: {time.report_elapsed(): .4} - Neurons: {pyMean([p.nT for p in pop])}")
        time.update() # Update time.
    else: # Load from checkpoints.
        gen, pop, hof, logbook, py_state, np_state = get_checkpoint(root="checkpoints", id_method=method,
                                                                    id_run=id_run)
        pySetstate(py_state) # Set python state.
        npSetstate(np_state) # Set numpy state.
        time = ExecutionTime(logbook.select("t")[-1]) # Get last run time value from logbook.
    bf = logbook.select("max")[-1] # Get the best fitness so far.
    gen += 1 # Update the generation.
    # ---- Main evolution loop:
    # Perform evolution only if there are still generations, and it hasn't reached a perfect individual.
    while gen <= gen_max and bf < 1.0:
        print('##############################################################################')
        # ---- Selection:
        offspring = toolbox.select(pop, pop_size) # Select individuals to generate offspring.
        offspring = list(map(toolbox.clone, offspring)) # Clone selected individuals.
        # ---- Crossover:
        for i, (child1, child2) in enumerate(zip(offspring[::2], offspring[1::2])):
            if cr >= pyRandom(): # Apply crossover with probability cr.
                toolbox.crossover(child1, child2) # Crossover operation.
                del child1.fitness.values, child2.fitness.values # Invalidate fitness value after crossover to re-evaluate it.
        # ---- Mutation:
        for mutant in offspring:
            if mr >= pyRandom(): # Apply mutation with probability mr.
                toolbox.mutate(mutant) # Mutation operation.
                del mutant.fitness.values # Invalidate fitness value after mutation to re-evaluate it.
        # ---- Evaluate offspring:
        new_individuals = [individual for individual in offspring if not individual.fitness.valid] # Get those individuals that had changed.
        fitting(toolbox, new_individuals) # Evaluate offspring that need it.
        print(f"Finished new evaluation. Elapsed time: {time.report_elapsed(): .4} - Num. Neurons {pyMean([p.nT for p in new_individuals])} - Num. Evaluated {len(new_individuals)}")
        # ---- Replacement:
        pop = toolbox.elitism(pop) + toolbox.replace(offspring) # Create new population with new individuals + elit from previous population.
        hof.update(pop)  # Update Hall of Fame with the current population.
        record_stats = stats.compile(pop)  # Collect statistics for the current population.
        logbook.record(gen=gen, t=time.report_run_time(), elapse_time=time.report_elapsed(),
                       evals=logbook.select("evals")[-1] + len(new_individuals), **record_stats)  # Update logbook.
        print(f"Generation {gen}: {record_stats}")
        print(f"Total run time: {logbook.select('t')[-1]: .4}")
        print(f"CPU status: %CPU: {psutil.cpu_percent()} - %RAM: {psutil.virtual_memory().percent}")
        time.update() # Update time.
        bf = logbook.select("max")[-1] # Get the best fitness so far.
        if gen % freq_check == 0: # Update checkpoints.
            save_checkpoint(root="checkpoints", id_method=method, id_run=id_run, gen=gen, pop=pop, hof=hof, log=logbook,
                            py_state=pyGetstate(), np_state=npGetstate())
        gen += 1 # Update generation
    # Close the multiprocessing pool to free resources, if necessary.
    if n_workers > 1:
        pool.close()
        pool.join()
    # ---- Validation:
    plot_options = {"convergence": False, "time": False, "liquid": False, "separability": False, "spikes": False,
                    "vms": False, "input": False} # Plot visualization options.
    acc_val = validation_sequence(data=data_val, labels=labels, population=pop, logbook=logbook, hof=hof,
                                  time_sim=sim_time, individual_on="Best", plot_options=plot_options)

    # Las checkpoints update.
    save_checkpoint(root="checkpoints", id_method=method, id_run=id_run, gen=gen, pop=pop, hof=hof, log=logbook,
                    py_state=pyGetstate(), np_state=npGetstate(), last=True, validation=acc_val)
    return bf, acc_val, logbook

if __name__ == '__main__':
    method = "GA_BLX_Perceptron-MCA2025"
    db = "FR5"
    initial_b = []
    final_b = []
    validation_b = []
    for i in range(30):
        print(f">>>>>>>>>>>> Run {i+1} <<<<<<<<<<<<")
        best_fitness, validation, lb = main(ds_name=db, id_method=f"{method}", id_run=f"{i+1}",
                                            n_workers=cpu_count())
        initial_b.append(lb[0]["max"])
        final_b.append(best_fitness)
        validation_b.append(validation[0])

    # Save summary
    with open(f"Results/{method}-{db}_Summary.data", "wb") as f:
        pickle.dump(obj={"Init_list": initial_b, "Init_mean": pyMean(initial_b), "Init_stdev": pyStdev(initial_b),
                         "Fin_list": final_b, "Fin_mean": pyMean(final_b), "Fin_stdev": pyStdev(final_b),
                         "Val_list": validation_b, "Val_mean": pyMean(validation_b), "Val_stdec": pyStdev(validation_b)},
                    file=f)
    print("----------- Summary -----------")
    print(f"Initial: {pyMean(initial_b)} +-{pyStdev(initial_b)}")
    print(f"Final: {pyMean(final_b)} +-{pyStdev(final_b)}")
    print(f"Validation: {pyMean(validation_b)} +-{pyStdev(validation_b)}")





