# General imports
from numpy import ndarray, array as npArray
from numpy.random import choice as npChoice
from sklearn.metrics import confusion_matrix
# NEST imposts
import nest
from nest.lib.hl_api_types import NodeCollection
from nest.lib.hl_api_models import CopyModel
from nest.lib.hl_api_nodes import Create
from nest.spatial.hl_api_spatial import free
from nest.lib.hl_api_connections import Connect
from nest.spatial_distributions import gaussian
from nest.spatial import distance
from nest.lib.hl_api_simulation import Simulate, ResetKernel, SetKernelStatus, GetKernelStatus
from nest.random.hl_api_random import normal
from nest.lib.hl_api_info import SetStatus
# DEAP imports
from deap import tools
# Source imports
from encoding import Encoding
from visualization import fitness_std, plot_input_data, plot_vm_separate, fitness_min, visualize_liquid, do_tsne, plot_spikes, exec_time
from auxiliary import spike_rate

# Level of verbosity for NEST kernel
nest.set_verbosity("M_ERROR")

class LSM:
    """
    Liquid State Machine class
    """
    # The following constants are necessary for inner procedures. They are not intended to be modified outside of it.
    neuron_model: str = 'iaf_psc_alpha'
    delay: float = 1.0  # This may be changed with future implementations

    def __init__(self, encoding: Encoding, inConn: float = 0.5, Ws: float = 50, neuronNoise_mean: float = 100.) -> None:
        # General parameters
        self.seed: int = encoding.lsm_seed  # NEST seed.
        self.inConn = inConn  # Input connectivity.
        self.neuronNoise_mean: float = neuronNoise_mean  # Standard deviation of the noise current amplitude (pA)
        self.neuronNoise_std: float = 2. # Standard deviation of the noise current amplitude (pA)
        self.Ws: float = Ws  # General strength for synapsis
        # TODO: Add window parameter to control the starting time for monitors to get information.
        # Topology parameters
        self.channels: int = encoding.channels
        self.indexesE: list[int] = encoding.indexesE
        self.indexesI: list[int] = encoding.indexesI
        self.nE: int = encoding.nE
        self.nI: int = encoding.nI
        self.nT: int = encoding.nT
        # Excitatory neuron parameters
        t_ref: list[float] = [encoding.configurations[i]["t_ref"] for i in self.indexesE]  # Duration of refractory period (ms).
        tau_m: list[float] = [encoding.configurations[i]["tau_m"] for i in self.indexesE]  # Membrane time constant (ms).
        v_th: list[float] = [encoding.configurations[i]["v_th"] for i in self.indexesE]  # Spike threshold (mV).
        v_reset: list[float] = [encoding._V_RESET if v > encoding._V_RESET else v - 5 for v in v_th]  # Reset potential of the membrane (mV).
        self.nE_conf: dict[str, float | list[float]] = {"t_ref": t_ref, "tau_m": tau_m, "V_th": v_th, "V_reset": v_reset, "E_L": -70.}
        # Inhibitory neuron parameters
        t_ref: list[float] = [encoding.configurations[i]["t_ref"] for i in self.indexesI]  # Duration of refractory period (ms).
        tau_m: list[float] = [encoding.configurations[i]["tau_m"] for i in self.indexesI]  # Membrane time constant (ms).
        v_th: list[float] = [encoding.configurations[i]["v_th"] for i in self.indexesI]  # Spike threshold (mV).
        v_reset: list[float] = [encoding._V_RESET if v > encoding._V_RESET else v - 5 for v in v_th]  # Reset potential of the membrane (mV).
        self.nI_conf: dict[str, float | list[float]] = {"t_ref": t_ref, "tau_m": tau_m, "V_th": v_th, "V_reset": v_reset, "E_L": -70.}
        # Positions
        self.nE_pos: list[list] = [encoding.positions[i] for i in self.indexesE]
        self.nI_pos: list[list] = [encoding.positions[i] for i in self.indexesI]
        # Initialized parameters
        self.inputC: NodeCollection | None = None
        self.monitor_vm: NodeCollection | None = None
        self.monitor_spikes: NodeCollection | None = None
        self.neuronsE: NodeCollection | None = None
        self.neuronsI: NodeCollection | None = None
        self.noiseG: NodeCollection | None = None

    def setup(self) -> None:
        """
        Setup and re-start some values. This is necessary since ResetKernel erase them.
        :return: None
        """
        self.inputC = None
        self.monitor_vm = None
        self.monitor_spikes = None
        self.neuronsE = None
        self.neuronsI = None
        CopyModel(existing='static_synapse', new='Input')
        CopyModel(existing='stdp_synapse', new='e_syn')
        CopyModel(existing='stdp_synapse', new='i_syn')
        CopyModel(existing='static_synapse', new='Monitor')

    def build(self) -> None:
        """
        Build the LSM (Decoding).
        :return: None
        """
        # Create input neuron. The content of the input channels will be given outside the method.
        self.inputC = Create(model="spike_generator", n=self.channels)
        # Create neuron populations
        self.neuronsE = Create(model=self.neuron_model, n=self.nE, params=self.nE_conf,
                               positions=nest.spatial.free(self.nE_pos))
        self.neuronsI = Create(model=self.neuron_model, n=self.nI, params=self.nI_conf,
                               positions=nest.spatial.free(self.nI_pos))
        self.noiseG = Create("noise_generator", params={"mean": self.neuronNoise_mean, "std": self.neuronNoise_std})
        # Create Monitors
        self.monitor_vm = Create(model="multimeter", params={"record_from": ["V_m"]})
        self.monitor_spikes = Create(model="spike_recorder")
        # TODO: Autapses is False since I am not sure if this actually affects.
        # --------------------> Inputs to excitatory: Out-degree
        Connect(self.inputC, self.neuronsE, conn_spec={"rule": "fixed_outdegree",
                                                       "outdegree": max(1, int(self.nE * self.inConn))},
                syn_spec={"synapse_model": "Input", "weight": normal(self.Ws, 2.)})
        # --------------------> Excitatory connections
        # To Excitatory
        Connect(self.neuronsE, self.neuronsE, conn_spec={"rule": "pairwise_bernoulli", "p": gaussian(distance),
                                                         'allow_autapses': False},
                syn_spec={"synapse_model": "e_syn", "weight": gaussian(distance) * self.Ws})
        # To Inhibitory
        Connect(self.neuronsE, self.neuronsI, conn_spec={"rule": "pairwise_bernoulli", "p": gaussian(distance),
                                                         'allow_autapses': False},
                syn_spec={"synapse_model": "e_syn", "weight": gaussian(distance) * self.Ws})
        # --------------------> Inhibitory connections
        # To Excitatory
        Connect(self.neuronsI, self.neuronsE, conn_spec={"rule": "pairwise_bernoulli", "p": gaussian(distance),
                                                         'allow_autapses': False},
                syn_spec={"synapse_model": "i_syn", "weight": gaussian(distance) * -2 * self.Ws})
        # --------------------> Noise connections
        Connect(self.noiseG, self.neuronsE)
        Connect(self.noiseG, self.neuronsI)
        # --------------------> Monitor Connections
        # Multimeter connects to all neurons. Spike recorder only to excitatory
        Connect(self.monitor_vm, self.neuronsE, syn_spec={'synapse_model': 'Monitor'})
        Connect(self.monitor_vm, self.neuronsI, syn_spec={'synapse_model': 'Monitor'})
        Connect(self.neuronsE, self.monitor_spikes, syn_spec={'synapse_model': 'Monitor'})

    def do_simulation(self, instances, time_sim) -> tuple[list, list]:
        """
        LSM simulation on spike train.
        :param instances: Input spike train.
        :param time_sim: Time of simulation. Not necessarily the same as spike_train length.
        :return: List of ndarray of LSM states & membrane voltage.
        """
        # Prepare the Kernel.
        ResetKernel()
        # Set LSM seed
        SetKernelStatus({"rng_seed": self.seed})
        # Load setup.
        self.setup()
        # Build LSM network.
        self.build()
        # Initialize empty list of recordables and biological time
        states = []
        vms = []
        biological_time = 0.
        # Iterate over instances
        for spike_train in instances:
            # Set spike trains to the channels
            SetStatus(self.inputC, [{"spike_times": spike_times + biological_time} for spike_times in spike_train])
            # Do Simulation
            Simulate(time_sim)
            # Get recordables
            spikes = self.monitor_spikes.get()
            vm = self.monitor_vm.get()
            # Correct registered times
            spikes["events"]["times"] -= spikes["origin"]
            vm["events"]["times"] -= vm["origin"]
            # Append values
            states.append(spikes)
            vms.append(vm)
            # Re-start Membrane Potential (Vm) for every neuron
            self.neuronsE.V_m = self.nE_conf["E_L"]
            self.neuronsI.V_m = self.nI_conf["E_L"]
            # Get current time
            biological_time = GetKernelStatus("biological_time")
            # Clean Monitors events
            self.monitor_vm.n_events = 0
            self.monitor_spikes.n_events = 0
            # Move the beginning of the Monitors
            self.monitor_vm.origin = biological_time
            self.monitor_spikes.origin = biological_time
            # Move the beginning of the noise generator
            self.noiseG.origin = biological_time
        return states, vms

def simulate_lsm(net: LSM, spike_train: list, time_sim: float) -> tuple[dict, dict]:
    """
    LSM simulation on spike train. Old implementation
    Deprecated. Only keeping it for historic reasons.
    :param net: LSM to be simulated.
    :param spike_train: Input spike train.
    :param time_sim: Time of simulation. Not necessarily the same as spike_train length.
    :return: List of ndarray of LSM states & membrane voltage.
    """
    # Prepare the Kernel.
    ResetKernel()
    # Set LSM seed
    SetKernelStatus({"rng_seed": net.seed})
    # Load setup.
    net.setup()
    # Build LSM network.
    net.build()
    # Add inputs to input channels.
    for i, in_channel in enumerate(net.inputC):
        in_channel.spike_times = spike_train[i]
    # Simulation
    Simulate(time_sim)
    # Retrieve information
    ### Membrane voltages
    mltm = net.monitor_vm.get()
    ### Spikes
    spikes = net.monitor_spikes.get()
    return spikes, mltm

def readout(X: ndarray, label: ndarray, classifier, train: bool = False, cm: bool = False) -> tuple:
    """
    Readout process
    :param X: States (Spike rate train)
    :param label: Labels
    :param classifier: Classifier. It must contain a fit method & a score method (Sklearn)
    :param train: If readout needs training or not
    :param cm: Print confusion matrix or not.
    :return: accuracy
    """
    if train:
        # Train classifier
        classifier.fit(X, label)
    # Get accuracy
    if cm:
        y_pred = classifier.predict(X)
        print(confusion_matrix(label, y_pred))

    accuracy: float = classifier.score(X, label)
    return accuracy, classifier

def validation(encoding: Encoding, lsm: LSM, data: dict, time_sim: float) -> tuple[float, ndarray, list, list]:
    """
    Validation process
    :param lsm: LSM net.
    :param data: Data.
    :param time_sim: Duration of simulation.
    :return: Accuracy, States, Spikes, Vms
    """
    # Reset NEST kernel
    ResetKernel()
    # Set the NEST seed
    SetKernelStatus({"rng_seed": lsm.seed})
    # Setup necessary configurations
    lsm.setup()
    # Build LSM
    lsm.build()
    #   State acquisition
    S_val, V_val = lsm.do_simulation(data["Spikes"], time_sim)
    X_val = []  # States: ndarray
    for s in S_val:
        rate = spike_rate(s, time_sim, lsm.nE, first_id=lsm.channels+1, window=0.)
        X_val.append(rate)
    X_val = npArray(X_val)
    #   Get Readout results
    acc_val,_ = readout(X_val, data["Labels"], encoding.classifier, train=False, cm=True)
    return acc_val, X_val, S_val, V_val

def validation_sequence(data: dict, labels: list[str], population: list, logbook: tools.Logbook, hof: tools.HallOfFame,
                        time_sim: float, individual_on: str | int = "Best", inConnectivity: float = 0.5,
                        plot_options: dict | None = None) -> None:
    """
    Validation Sequence:
        1. Plots convergence plot.
        2. Plot execution time per generation.
        3. Probe Best (Random or specific) individual (LSM) on test set.
        4. Do Separability (T-SNE) visualization.
        5. Plot Inputs, Vms and spikes for each class from a random instance.
    :param data: Data used for validation.
    :param labels: List of string for labels.
    :param population: DEAP population.
    :param logbook: DEAP Logbook.
    :param hof: DEAP HallOfFame.
    :param time_sim: Duration of simulation.
    :param individual_on: Decision on what individual validate. Options are:
            1. Best: Validation on HOF
            2. Random: Validation on random individual from population
            3. Specific individual (int): Validation on a specific individual from population. Must ensure that the number mathc the length of population
    :param inConnectivity: Percentage of Input-Excitatory connectivity.
    :param plot_options: Options for plotting: "convergence", "time", "liquid", "separability", "input", "spikes", "vms". None means all plots.
    :return: None
    """
    # Default plot options if none are provided
    if plot_options is None:
        plot_options = {"convergence": True, "time": True,  "liquid": True, "separability": True, "input": True,
                        "spikes": True, "vms": True}
    # 1. Plot convergence plot
    #   Getting information from logbook
    gens: list = logbook.select("gen")
    fitness: list = logbook.select("max")
    mean_f: list = logbook.select("avg")
    std_f: list = logbook.select("std")
    min_f: list = logbook.select("min")
    #   Plot
    if plot_options["convergence"]:
        fitness_std(gens, fitness, mean_f, std_f)
        fitness_min(gens, fitness, mean_f, min_f)
    # 2. Plot execution time
    #   Getting information from logbook
    exec_t: list = logbook.select("t")
    elapsed_t: list = logbook.select("e_t")
    #   Plot
    if plot_options["time"]:
        exec_time(gens, exec_t, elapsed_t)
    # 3. Probe Best (Random or specific) individual (LSM) on test set.
    encoding: Encoding
    #   Check decision on individual to validate
    if isinstance(individual_on, str):
        match individual_on:
            # Best individual: HOF
            case "Best":
                encoding = hof[0]
            case "Random":
                selection = npChoice(len(population))
                print(f"Selecting LSM: {selection} from population")
                encoding = population[selection]
            case _:
                print("Option not recognized")
    elif isinstance(individual_on, int):
        # Ensure that int is within ranged
        if 0>= individual_on < len(population):
            encoding = population[individual_on]
        else:  # int is not within ranged, a random valid number will be drawn
            selection = npChoice(len(population))
            encoding = population[selection]
    #   Build LSM
    lsm = LSM(encoding, inConn=inConnectivity)
    #   Do validation
    acc, states, spikes, vms = validation(encoding, lsm, data, time_sim)
    #   Show LSM
    if plot_options["liquid"]:
        visualize_liquid(lsm)
    print(f"Validation accuracy: {acc}")
    # 4. Do Separability (T-SNE) visualization.
    if plot_options["separability"]:
        do_tsne(states, labels, name="Validation (T-SNE)")
    # 5. Plot Inputs, Vms and spikes for each class from a random instance.
    #   Get random section
    blocks = int(len(data["Labels"]) / len(labels))
    pos = npChoice(blocks) * len(labels)
    #   Select data and plot
    for i, c in enumerate(labels):
        if plot_options["input"]:
            plot_input_data(data["Spikes"][i+pos], name=f"Class {c}: Input")
        if plot_options["spikes"]:
            plot_spikes(spikes[i], lsm.nE, lsm.channels+1, name=f"Class {c}: Spikes")
        if plot_options["vms"]:
            plot_vm_separate(vms[i], lsm, name=f"Class {c}: Membrane Potential per Neuron")
            plot_vm_separate(vms[i], lsm, name=f"Class {c}: Membrane Potential per Neuron with Threshold values",
                             w_threshold=True)

if __name__ == '__main__':
    """ Test on differences between simulations"""
    import pickle
    from auxiliary import split_data
    import matplotlib.pyplot as plt
    from random import seed as  pySeed
    from numpy.random import seed as npSeed

    import time

    classes = 2
    labels = ["C1", "C2"]
    # labels = ["C1", "C2", "C3", "C4", "C5"]
    channels = 3
    n_neurons = 20
    sim_time = 110.
    with open(f"../Experiments/Datasets/binary_test_small.data", "rb") as file:
        data = pickle.load(file)

    data_train, data_val = split_data(classes, data, 0.7, 0.6)
    pySeed(0)
    npSeed(0)
    encoding = Encoding(n_neurons, channels)
    lsm = LSM(encoding)
    t1 = []
    for _ in range(50):
        start = time.time()
        r = lsm.do_simulation(data_train["Spikes"], 110.)
        t1.append(time.time() - start)
    plt.figure(1)
    plt.plot(t1)
    plt.show()
    time.sleep(1)

    for i, c in enumerate(labels):
        plot_spikes(r[0][0], lsm.nE, lsm.channels + 1, name=f"C{c}: Spikes")
        plot_vm_separate(r[1][-1], lsm, name=f"C{c}: Vms")
    t2 = []
    for _ in range(50):
        X: list = []
        vms = []
        start = time.time()
        for i, x in enumerate(data_train["Spikes"]):
            spikes_dict, vm = simulate_lsm(lsm, x, 110.)
            X.append(spikes_dict)
            vms.append(vm)
        t2.append(time.time() - start)
    plt.figure(2)
    plt.plot(t2)
    plt.show()

    for i, c in enumerate(labels):
        plot_spikes(X[0], lsm.nE, lsm.channels + 1, name=f"C{c}: Spikes")
        plot_vm_separate(vms[-1], lsm, name=f"C{c}: Vms")
