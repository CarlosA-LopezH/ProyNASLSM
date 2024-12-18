# General imports
from numpy.random import random as npRandom, randint as npRandInt, set_state as npSetstate, get_state as npGetstate, choice as npChoice, rand as npRand
from random import uniform
from math import ceil
import pickle
from multiprocessing import cpu_count, Pool
from sklearn.linear_model import Perceptron
from random import setstate as pySetstate, random as pyRandom, getstate as pyGetstate, uniform as pyUniform
from numpy import mean as npMean, std as npStd, min as npMin, max as npMax, ndarray
from pathlib import Path
from statistics import mean as pyMean, stdev as pyStdev
from random import sample
# DEAP imports
from deap import base, creator, tools
# Source imports
from lib.operationsEvo import evaluation, fitting, get_checkpoint, save_checkpoint
from lib.auxiliary import ExecutionTime, split_data
from lib.LSM import validation_sequence

class Enc_Position:
    """
    Encoding method. It contains only the configurations of neurons.
    """
    # The following constants are necessary for inner procedures. They are not intended to be modified outside of it.
    # The original values are at the middle of the options. These are not based on any reference.
    _T_REF: list[float] = [0.5, 1., 2., 3., 4.]  # Duration of refractory period (ms).
    _TAU_M: list[float] = [6., 8., 10., 12., 14.]  # Membrane time constant (ms).
    _V_TH: list[float] = [-69., -55., -40., -25., -10.]  # Spike threshold (mV).
    _V_RESET: float = -70.  # Reset potential of the membrane (mV).
    _PROB_E: float = 0.8  # Probability for setting the excitatory polarity. Based on "Neuron Dynamics"
    def __init__(self, n_neurons: int, channels: int, dim: float = 6.) -> None:
        # General parameters
        self.dim: float = ceil(dim)  # The size (dimension) of the space for neuron positions. With bigger dimension, the distances grow
        self.channels = channels  # Number of inputs
        self.channels = channels  # Number of inputs
        self.nT: int = n_neurons  # Total number of neurons
        self.nE: int = 0  # Number of excitatory neurons
        self.nI: int = 0  # Number of inhibitory neurons
        # Initialize encoding elements
        self.configurations: list[dict] = []
        self.positions: list[list] = []
        # Random encoding
        for _ in range(self.nT):
            self.configurations.append(self.set_configuration())
            self.positions.append(self.set_positions())
        # Initialize the indexes for excitatory and inhibitory, and the sizes
        self.indexesE: list | None = None
        self.indexesI: list | None = None
        self.polarity_indexing()
        # Set seed for all individuals.
        self.lsm_seed: int = 0
        self.set_seed()
        # Set variable for classifier. It will be assigned after evaluation.
        self.classifier = None

    def set_configuration(self) -> dict[str, float | str]:
        """
        Generates a random dictionary with configurations.
        :return: dictionary with configurations.
        """
        # Build configuration
        configuration = {'t_ref': round(uniform(self._T_REF[0], self._T_REF[-1]), 2),
                         'tau_m': round(uniform(self._TAU_M[0], self._TAU_M[-1]), 2),
                         'v_th': round(uniform(self._V_TH[0], self._V_TH[-1]), 2),
                         'polarity': 'E' if npRandom() < self._PROB_E else 'I'}
        return configuration

    def set_positions(self) -> list[float]:
        """
        Generated a random x, y position. To be valid, no other neuron has to have the same position.
        :return: list of x, y positions.
        """
        # Flag to force valid (no repeated) position.
        invalid: bool = True
        # Initialize coordinates.
        x: float = 0.
        y: float = 0.
        # Find valid position.
        while invalid:
            x = round(uniform(0, self.dim), 2)  # X position.
            y = round(uniform(0, self.dim), 2)  # Y position.
            invalid = [x, y] in self.positions
        return [x, y]

    def set_seed(self) -> None:
        """ Generates random seed for NEST simulator"""
        self.lsm_seed = npRandInt(1, 100000)

    def polarity_indexing(self) -> None:
        """
        Obtain the indexes based on the polarity of the neurons.
        :return: Excitatory & Inhibitory indexes.
        """
        # Initialize list of indexes for neurons.
        exc_indexes: list[int] = []
        inh_indexes: list[int] = []
        # Iterate over neuron configurations.
        for i, e in enumerate(self.configurations):
            if e['polarity'] == 'E':
                exc_indexes.append(i)
            else:
                inh_indexes.append(i)
        # In rare occasions, there are few (<=1) excitatory & inhibitory neurons. This leads to some errors.
        # To solve this, a new neuron is created, forcing its polarity.
        while len(exc_indexes) <= 1 or len(inh_indexes) <= 1:
            # Checking the excitatory condition
            if len(exc_indexes) <= 1:
                # Obtain a random configuration.
                exc_conf = self.set_configuration()
                # Force polarity.
                exc_conf['polarity'] = 'E'
                # Add neuron configuration to the encoding.
                self.configurations.append(exc_conf)
                # The new neuron needs a position.
                # Obtain a position.
                exc_pos = self.set_positions()
                # Add neuron position to the encoding.
                self.positions.append(exc_pos)
                # Since the new neuron corresponds to the last append, the (size - 1) of the configurations is added
                # as index.
                exc_indexes.append(len(self.configurations) - 1)
            # Checking the inhibitory condition
            if len(inh_indexes) <= 1:
                # Obtain a random configuration.
                inh_conf = self.set_configuration()
                # Force polarity.
                inh_conf['polarity'] = 'I'
                # Add neuron configuration to the encoding.
                self.configurations.append(inh_conf)
                # Obtain a position.
                inh_pos = self.set_positions()
                # Add neuron position to the encoding.
                self.positions.append(inh_pos)
                # Since the new neuron corresponds to the last append, the (size - 1) of the configurations is added
                # as index.
                inh_indexes.append(len(self.configurations) - 1)
        # Upate indexes and sizes
        self.indexesE = exc_indexes
        self.indexesI = inh_indexes
        self.nE = len(exc_indexes)
        self.nI = len(inh_indexes)
        self.nT = self.nE + self.nI

def initEncoding(encoding: Enc_Position, dimensions: tuple | int, channels: int) -> Enc_Position:
    # Check if dimensions is a range or just one number (fixed size)
    if isinstance(dimensions, tuple):
        n_neurons: int = npRandInt(dimensions[0], dimensions[1])
    else:
        n_neurons: int = dimensions
    return encoding(n_neurons, channels)

def xover_blx(parent1: Enc_Position, parent2: Enc_Position, alpha: float = 0.5) -> tuple[Enc_Position, Enc_Position]:
    """
    Crossover operation: Adaptation of Blend crossover.
    :param parent1: Parent 1.
    :param parent2: Parent 2.
    :param alpha: Value of variation.
    :return: Offspring
    """
    # Get keys: Configurations to be considered.
    keys = parent1.configurations[0].keys()
    # Child 1 is always the smaller.
    if parent1.nT <= parent2.nT:
        child1, child2 = parent1, parent2
    else:
        child1, child2 = parent2, parent1
    # Recombination order.
    order1 = sample(range(child1.nT), child1.nT)
    order2 = sample(range(child2.nT), child1.nT)
    # Do crossing.
    for i, j in zip(order1, order2):
        ### Recombination on positions
        for h, (v1, v2) in enumerate(zip(child1.positions[i], child2.positions[j])):
            lower = min(v1, v2)
            upper = max(v1, v2)
            d = upper - lower
            # Establish bounds
            lower -= d * alpha
            # Restriction: No coordinate can be lower than 0
            if lower < 0.:
                lower = 0.
            upper += d * alpha
            # Update position
            child1.positions[i][h] = round(uniform(lower, upper), 2)
            child2.positions[j][h] = round(uniform(lower, upper), 2)
        # Update encodings: Child 1
        child1.polarity_indexing()
        child1.set_seed()
        child1.classifier = None
        # Update encodings: Child 2
        child2.polarity_indexing()
        child2.set_seed()
        child2.classifier = None
    return child1, child2

def mutation_simple(individual: Enc_Position) -> Enc_Position:
    """
    Mutation. Three types of mutation operation:
        Mutation 1. Reset of some neuron positions.
    :param individual: Individual to be mutated.
    :return: Mutated individual.
    """
    # Mutation size: Number of neurons to be mutated. The value is maximum the half size of the network.
    mut_size: int = int((individual.nT / 2) * npRand())
    # Indexes of the neurons to be mutated.
    mut_neurons: ndarray = npChoice(range(individual.nT), mut_size, replace=False)
    for id_neuron in mut_neurons:
        individual.positions[id_neuron] = individual.set_positions()
    # Update encoding
    individual.polarity_indexing()
    individual.set_seed()
    individual.classifier = None
    return individual

# Create fitness and individual classes.
creator.create("Fitness", base.Fitness, weights=(1.0,))  # Fitness maximization.
creator.create("Individual", Enc_Position, fitness=creator.Fitness) # Individual class based on encoding.

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
    pop_size: int = 20 # Population size.
    gen_max: int = 100 # Max number of generations.
    t_size: int = 5 # Tournament size.
    cr: float = 0.8 # Crossover rate.
    mr: float = 0.5 # Mutation rate.
    elitism: int = 1 # Number of individuals to be preserved.
    # ---- LSM Parameters:
    channels: int = data["Channels"] # Number of input channels.
    sim_time: int = data["Tmax"] + 10 # Duration of NEST simulation.
    net_size: int = 20 # Number of neurons in the liquid.
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
    # Check if a checkpoint exists to resume the evolution.
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
    else: # Load from checkpoint.
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
                       evals=len(new_individuals), **record_stats)  # Update logbook.
        print(f"Generation {gen}: {record_stats}")
        print(f"Total run time: {logbook.select('t')[-1]: .4}")
        time.update() # Update time.
        bf = logbook.select("max")[-1] # Get the best fitness so far.
        if gen % freq_check == 0: # Update checkpoint.
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
    acc_val = validation_sequence(data=data_val, labels=labels, population=pop, logbook=logbook, hof=hof, time_sim=sim_time,
                                  individual_on="Best", plot_options=plot_options)
    # Las checkpoint update.
    save_checkpoint(root="checkpoints", id_method=method, id_run=id_run, gen=gen, pop=pop, hof=hof, log=logbook,
                    py_state=pyGetstate(), np_state=npGetstate(), last=True, validation=acc_val)
    return bf, acc_val, logbook

if __name__ == '__main__':
    db = "FR5"
    method = "GA_BLX_Perceptron-GECCO2025_Positions"
    initial_b = []
    final_b = []
    validation_b = []
    for i in range(30):
        print(f">>>>>>>>>>>> Run {i + 1} <<<<<<<<<<<<")
        best_fitness, validation, lb = main(ds_name=db, id_method=f"{method}", id_run=f"{i + 1}",
                                            n_workers=cpu_count())
        initial_b.append(lb[0]["max"])
        final_b.append(best_fitness)
        validation_b.append(validation[0])

    # Save summary
    with open(f"Results/{method}_Summary.data", "wb") as f:
        pickle.dump(obj={"Init_list": initial_b, "Init_mean": pyMean(initial_b), "Init_stdev": pyStdev(initial_b),
                         "Fin_list": final_b, "Fin_mean": pyMean(final_b), "Fin_stdev": pyStdev(final_b),
                         "Val_list": validation_b, "Val_mean": pyMean(validation_b),
                         "Val_stdec": pyStdev(validation_b)},
                    file=f)
    print("----------- Summary -----------")
    print(f"Initial: {pyMean(initial_b)} +-{pyStdev(initial_b)}")
    print(f"Final: {pyMean(final_b)} +-{pyStdev(final_b)}")
    print(f"Validation: {pyMean(validation_b)} +-{pyStdev(validation_b)}")