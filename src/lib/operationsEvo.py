# General imports
import pickle
from numpy.random import randint as npRandint, rand as npRand, choice as npChoice
from numpy import ndarray, array as npArray
from sklearn.neighbors import KNeighborsClassifier as KNN
from random import sample, uniform
# Source imports
from .encoding import Encoding
from .LSM import LSM, readout
from .auxiliary import spike_rate


# ----------------------------------------------------- #
#                       Initialization                  #
# ----------------------------------------------------- #
def initEncoding(encoding: Encoding, dimensions: tuple | int, channels: int) -> Encoding:
    # Check if dimensions is a range or just one number (fixed size)
    if isinstance(dimensions, tuple):
        n_neurons: int = npRandint(dimensions[0], dimensions[1])
    else:
        n_neurons: int = dimensions
    return encoding(n_neurons, channels)

# ----------------------------------------------------- #
#                       Evaluation                      #
# ----------------------------------------------------- #
def fitting(toolbox, eval_pop) -> None:
    """
    Evaluates the fitness of the individuals in the population using the toolbox.
    :param toolbox: DEAP toolbox containing evaluation function.
    :param eval_pop: Population to be evaluated.
    """
    eval_result = toolbox.map(toolbox.evaluate, eval_pop)  # Parallel evaluation using pool
    for ind, result in zip(eval_pop, eval_result):
        ind.fitness.values = result[0]  # Assigns the evaluated fitness values to each individual
        ind.classifier = result[1]

def evaluation(individual: Encoding, inputs: dict, time_sim: float, classifier = KNN, window=1.) -> tuple:
    """
    Evaluation of individuals.
    :param individual: Individual to be evaluated.
    :param inputs: Input spike train.
    :param time_sim: Time of simulation. Not necessarily the same as spike_train length.
    :param window: Percentage of time to retrieve states.
    :param classifier:
    :return: Accuracy
    """
    # Unpack labels
    labels: ndarray = inputs["Labels"]
    # Get LSM
    lsm = LSM(individual)
    # Get the time to get states
    windowing = time_sim - (window * time_sim)
    # Get states
    X: list = []
    spikes_dicts, _ = lsm.do_simulation(inputs["Spikes"], time_sim)
    for spikes_dict in spikes_dicts:
        states = spike_rate(spikes_dict, time_sim, lsm.nE, first_id=individual.channels+1, window=windowing)
        X.append(states)
    # Split on training and testing
    x_train = npArray(X[:inputs["Point"]])
    x_test = npArray(X[inputs["Point"]:])
    if x_test.shape[0] > 0:
        train_acc, classifier = readout(x_train, labels[:inputs["Point"]], classifier(), train=True)
        # print(f"Train Accuracy: {train_acc}")
        acc, _ = readout(x_test, labels[inputs["Point"]:], classifier, train=False)
    else:
        acc, classifier = readout(x_train, labels, classifier(), train=True)
    return (acc,), classifier

# ----------------------------------------------------- #
#                       Crossover                       #
# ----------------------------------------------------- #
def xover_1pto(parent1: Encoding, parent2: Encoding) -> tuple[Encoding, Encoding]:
    """
    Crossover operation: One-point crossover.
    :param parent1: Parent 1.
    :param parent2: Parent 2.
    :return: Offspring
    """
    # Child 1 is always the smaller.
    if parent1.nT <= parent2.nT:
        child1, child2 = parent1, parent2
    else:
        child1, child2 = parent2, parent1
    # Select point of crossing.
    k = npRandint(1, child1.nT)
    # Do crossing.
    child1.configurations[k:], child2.configurations[k:] = child2.configurations[k:], child1.configurations[k:]
    child1.positions[k:], child2.positions[k:] = child2.positions[k:], child1.positions[k:]
    # Update encodings: Child 1
    child1.polarity_indexing()
    child1.set_seed()
    child1.classifier = None
    # Update encodings: Child 2
    child2.polarity_indexing()
    child2.set_seed()
    child2.classifier = None
    return child1, child2

def xover_blx(parent1: Encoding, parent2: Encoding, alpha: float = 0.5) -> tuple[Encoding, Encoding]:
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
        ### Recombination on configurations
        for key in keys:
            if key != 'polarity':  # For polarity, every child inherit the parent's
                lower = min(child1.configurations[i][key], child2.configurations[j][key])
                upper = max(child1.configurations[i][key], child2.configurations[j][key])
                d = upper - lower
                # Establish bounds
                lower -= d * alpha
                # Restriction: T_Ref and Tau_M cannot be lower than 0.
                if key == 't_ref' or key == 'tau_m' and lower <= 0.:
                    lower = 0.01
                upper += d * alpha
                # Update configurations
                child1.configurations[i][key] = round(uniform(lower, upper), 2)
                child2.configurations[j][key] = round(uniform(lower, upper), 2)
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
        # Update encodings: Child 1
        child2.polarity_indexing()
        child2.set_seed()
        child2.classifier = None
    return child1, child2

def xover_uniform(parent1: Encoding, parent2: Encoding, probability: float = 0.5) -> tuple[Encoding, Encoding]:
    """
    Crossover operation: Adaptation of Uniform crossover. This method is applied only if the size of parents is equal.
    :param parent1: Parent 1.
    :param parent2: Parent 2.
    :param probability: Value of variation.
    :return: Offspring
    """
    # Prepare encoding for both offspring
    config_c1 = []
    pos_c1 = []
    config_c2 = []
    pos_c2 = []
    # Iterate over neurons
    for i in range(parent1.nT):
        # Randomly assign the inheritance of the neuron.
        if npRand() <= probability:
            # Child 1 receives the neuron from parent 1.
            config_c1.append(parent1.configurations[i])
            pos_c1.append(parent1.positions[i])
            # Child 2 receives the neuron from parent 2.
            config_c2.append(parent2.configurations[i])
            pos_c2.append(parent2.positions[i])
        else:
            # Child 1 receives the neuron from parent 2.
            config_c1.append(parent2.configurations[i])
            pos_c1.append(parent2.positions[i])
            # Child 2 receives the neuron from parent 1.
            config_c2.append(parent1.configurations[i])
            pos_c2.append(parent1.positions[i])
    # Generate both children and assign their configurations and positions.
    child1, child2 = parent1, parent2
    child1.configurations = config_c1
    child1.positions = pos_c1
    child2.configurations = config_c2
    child2.positions = pos_c2
    # Update encoding: Child 1
    child1.polarity_indexing()
    child1.set_seed()
    child1.classifier = None
    # Update encoding: Child 1
    child2.polarity_indexing()
    child2.set_seed()
    child2.classifier = None
    return child1, child2
# ----------------------------------------------------- #
#                       Mutation                        #
# ----------------------------------------------------- #
def mutation(individual: Encoding) -> Encoding:
    """
    Mutation. Three types of mutation operation:
        Mutation 1. Reset of some neuron configurations.
        Mutation 2. Reset of some neuron positions.
        Mutation 3. Add/Delete neurons.
    :param individual: Individual to be mutated.
    :return: Mutated individual.
    """
    # Mutation probability.
    mut_value: float = npRand()
    # Mutation size: Number of neurons to be mutated. The value is maximum the half size of the network.
    mut_size: int = int((individual.nT / 2) * npRand())
    # Indexes of the neurons to be mutated.
    mut_neurons: ndarray = npChoice(range(individual.nT), mut_size, replace=False)
    if 0.6 <= mut_value:  # Mutation 1.
        for id_neuron in mut_neurons:
            individual.configurations[id_neuron] = individual.set_configuration()
    elif 0.3 <= mut_value:  # Mutation 2.
        for id_neuron in mut_neurons:
            individual.positions[id_neuron] = individual.set_positions()
    else:  # Mutation 3.
        if npRand() <= 0.5:  # Add neurons.
            for _ in range(mut_size):
                individual.configurations.append(individual.set_configuration())
                individual.positions.append(individual.set_positions())
        else:  # Delete neurons.
            individual.configurations = [config for i, config in enumerate(individual.configurations) if i not in mut_neurons]
            individual.positions = [pos for i, pos in enumerate(individual.positions) if i not in mut_neurons]
    # Update encoding
    individual.polarity_indexing()
    individual.set_seed()
    individual.classifier = None
    return individual

def mutation_simple(individual: Encoding) -> Encoding:
    """
    Mutation. Three types of mutation operation:
        Mutation 1. Reset of some neuron configurations.
        Mutation 2. Reset of some neuron positions.
    :param individual: Individual to be mutated.
    :return: Mutated individual.
    """
    # Mutation probability.
    mut_value: float = npRand()
    # Mutation size: Number of neurons to be mutated. The value is maximum the half size of the network.
    mut_size: int = int((individual.nT / 2) * npRand())
    # Indexes of the neurons to be mutated.
    mut_neurons: ndarray = npChoice(range(individual.nT), mut_size, replace=False)
    if 0.5 <= mut_value:  # Mutation 1.
        for id_neuron in mut_neurons:
            individual.configurations[id_neuron] = individual.set_configuration()
    else:  # Mutation 2.
        for id_neuron in mut_neurons:
            individual.positions[id_neuron] = individual.set_positions()
    # Update encoding
    individual.polarity_indexing()
    individual.set_seed()
    individual.classifier = None
    return individual

# ----------------------------------------------------- #
#                       Checkpoint                      #
# ----------------------------------------------------- #
def get_checkpoint(root: str, id_method: str, id_run: str) -> tuple:
    """
    Recover checkpoints
    :param root: Root directory to retrieve checkpoints.
    :param id_method: ID of the method.
    :param id_run: ID of the run.
    :return:
    """
    with open(f"{root}/{id_method}_{id_run}.chck", "rb") as file:
        checkpoint = pickle.load(file)
    gen = checkpoint["Generation"]
    pop = checkpoint["Population"]
    hof = checkpoint["HallOfFame"]
    logbook = checkpoint["Logbook"]
    pyState = checkpoint["pyRandomState"]
    npState = checkpoint["npRandomState"]
    print(f"Loading checkpoints {id_method}_{id_run}.chck")
    print(f"Current run time: {logbook.select('t')[-1]}")
    print(f"Starting from generation: {gen}")
    return gen, pop, hof, logbook, pyState, npState

def save_checkpoint(root: str, id_method: str, id_run: str, gen, pop, hof, log, py_state, np_state, last: bool = False) -> None:
    """
    Save checkpoints
    :param root: Root directory to retrieve checkpoints.
    :param id_method: ID of the method.
    :param id_run: ID of the run.
    :param gen: Current generation.
    :param pop: Population
    :param hof: HallOfFame
    :param log: Logbook
    :param py_state: Python random state.
    :param np_state: Numpy random state.
    :param last: Flag to indicate if this is the last checkpoints.
    :return:
    """
    checkpoint = {"Generation": gen,
                  "Population": pop,
                  "HallOfFame": hof,
                  "Logbook": log,
                  "pyRandomState": py_state,
                  "npRandomState": np_state}
    with open(f"{root}/{id_method}_{id_run}.chck", "wb") as file:
        pickle.dump(checkpoint, file)
    if not last:
        print(f"Saved checkpoints for generation: {gen}!")
    else:
        print(f"Saved last checkpoints!")







