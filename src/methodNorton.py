from statistics import mean

import numpy as np
import random
from nest.lib.hl_api_nodes import Create
import nest
from nest.lib.hl_api_connections import Connect
from nest.lib.hl_api_models import CopyModel
from nest.math import redraw
from nest.spatial_distributions import gaussian
from nest.lib.hl_api_simulation import Simulate, ResetKernel, SetKernelStatus, GetKernelStatus
from nest.random.hl_api_random import normal
from nest.lib.hl_api_info import SetStatus

from copy import deepcopy
from sklearn.linear_model import Perceptron

nest.set_verbosity("M_ERROR")
# nest.resolution = 0.001

# Utility: Compute centers of mass for all classes
def compute_centers_of_mass(state_vectors, class_labels):
    """
    Compute centers of mass for each class.
    Parameters:
        state_vectors (np.ndarray): Matrix of state vectors (rows = samples, cols = neurons).
        class_labels (np.ndarray): Array of class labels corresponding to state vectors.
    Returns:
        dict: A dictionary where keys are class labels and values are the centers of mass.
    """
    classes = np.unique(class_labels)
    return {cls: np.mean(state_vectors[class_labels == cls], axis=0) for cls in classes}

# Inter-class Distance
def inter_class_distance(centers_of_mass):
    """
    Calculate inter-class distance (cd) based on centers of mass.
    Parameters:
        centers_of_mass (dict): Dictionary of class centers of mass.
    Returns:
        float: Inter-class distance.
    """
    class_centers = list(centers_of_mass.values())
    n_classes = len(class_centers)
    cd = sum(np.linalg.norm(class_centers[i] - class_centers[j]) for i in range(n_classes) for j in range(n_classes)) / n_classes**2
    return cd

# Intra-class Variance
def intra_class_variance(state_vectors, class_labels, centers_of_mass):
    """
    Calculate intra-class variance (cv) for a given set of state vectors.
    Parameters:
        state_vectors (np.ndarray): Matrix of state vectors (rows = samples, cols = neurons).
        class_labels (np.ndarray): Array of class labels corresponding to state vectors.
        centers_of_mass (dict): Dictionary of class centers of mass.
    Returns:
        float: Intra-class variance.
    """
    total_variance = 0
    for cls, center_of_mass in centers_of_mass.items():
        class_vectors = state_vectors[class_labels == cls]
        variance = np.mean(np.linalg.norm(class_vectors - center_of_mass, axis=1))
        total_variance += variance
    return total_variance / len(centers_of_mass)

# Separation Metric
def separation_metric(state_vectors, class_labels):
    """
    Calculate separation (SepC) for a given set of state vectors.
    Parameters:
        state_vectors (np.ndarray): Matrix of state vectors (rows = samples, cols = neurons).
        class_labels (np.ndarray): Array of class labels corresponding to state vectors.
    Returns:
        float: Separation metric.
    """
    centers_of_mass = compute_centers_of_mass(state_vectors, class_labels)
    cd = inter_class_distance(centers_of_mass)
    cv = intra_class_variance(state_vectors, class_labels, centers_of_mass)
    return cd / (cv + 1)  # Add 1 to cv to avoid division by zero

# Select Random Samples Per Class
def select_random_samples(samples, class_labels, n_samples=3):
    """
    Select n_samples random state vectors per class.
    Parameters:
        samples (np.ndarray): Matrix of state vectors (rows = samples, cols = neurons).
        class_labels (np.ndarray): Array of class labels corresponding to state vectors.
        n_samples (int): Number of samples to select per class.

    Returns:
        np.ndarray, np.ndarray: Reduced state vectors and corresponding class labels.
    """
    classes = np.unique(class_labels)
    selected_vectors = []
    selected_labels = []
    for cls in classes:
        class_vectors = []
        for i, decision in enumerate(class_labels == cls):
            if decision:
                class_vectors.append(samples[i])
        # class_vectors = samples[class_labels == cls]
        if len(class_vectors) > n_samples:
            indices = random.sample(range(len(class_vectors)), n_samples)
            for ind in indices:
                selected_vectors.append(class_vectors[ind])
            selected_labels.extend([cls] * n_samples)
        else:
            selected_vectors.append(class_vectors)
            selected_labels.append([cls] * len(class_vectors))
    return selected_vectors, np.array(selected_labels)

# Neuron Activity and Variances
def calculate_neuron_activity_and_variances(state_vectors, class_labels):
    """
    Calculate neuron activity and variances for each neuron.
    Parameters:
        state_vectors (np.ndarray): Matrix of state vectors (rows = samples, cols = neurons).
        class_labels (np.ndarray): Array of class labels corresponding to state vectors.

    Returns:
        tuple: (activity, variances), where both are 1D arrays.
    """
    centers_of_mass = compute_centers_of_mass(state_vectors, class_labels)
    classes = np.unique(class_labels)
    n_neurons = state_vectors.shape[1]
    # Neuron activity
    mean_state_vectors = [centers_of_mass[cls] for cls in classes]
    activity = np.mean(mean_state_vectors, axis=0)
    # Neuron variances
    neuron_variances = np.zeros(n_neurons)
    for cls in classes:
        class_vectors = state_vectors[class_labels == cls]
        center_of_mass = centers_of_mass[cls]
        neuron_variances += np.mean(np.abs(class_vectors - center_of_mass), axis=0)
    neuron_variances /= len(classes)
    return activity, neuron_variances

# Generate Artificial State Vectors and Calculate Sep*
def calculate_sep_star(n_classes, n_neurons, n_samples=3):
    """
    Generate artificial state vectors and calculate Sep*.
    Parameters:
        n_classes (int): Number of classes.
        n_neurons (int): Number of neurons per state vector.
        n_samples (int): Number of artificial state vectors per class.
    Returns:
        float: The calculated Sep* value.
    """
    state_vectors = []
    class_labels = []
    # Generate one unique center for each class, maximally separated
    class_centers = np.eye(n_classes, n_neurons)[:n_classes]  # Identity matrix-like centers
    for cls, center in enumerate(class_centers):
        for _ in range(n_samples):
            # Add small random noise to center to create artificial state vector
            jittered_vector = center + np.random.normal(0, 0.01, size=n_neurons)
            state_vectors.append(jittered_vector)
            class_labels.append(cls)
    state_vectors = np.array(state_vectors)
    class_labels = np.array(class_labels)
    # Calculate Sep*
    return separation_metric(state_vectors, class_labels)

# Update Function e(t)
def compute_e(w_ij, cd, max_separation, neuron_activity, neuron_variance, mu_w, mw):
    """
    Calculate the synaptic update factor e(t).
    Parameters:
        w_ij (float): Current weight of the synapse.
        cd (float): Inter-class distance.
        max_separation (float): Approximate maximum separation for the problem.
        neuron_activity (float): Activity of the post-synaptic neuron.
        neuron_variance (float): Variance of the neuron's activity across classes.
        mu_w (float): Expected value of the magnitude of synaptic weights in the initial liquid.
        mw (float): Estimate of the maximum of the random values drawn from the initialization distribution.
    Returns:
        float: Update factor e(t).
    """
    relative_strength = (abs(w_ij) - mu_w) / mw  # rs calculation
    di = neuron_activity * (1 - cd / max_separation)  # Distance correction
    vi = neuron_activity * neuron_variance  # Variance correction
    return relative_strength * (vi - di)

# Activity Function f(t)
def compute_f(global_activity, weight, e_t, k=6):
    """
    Calculate the activity-based adjustment f(t).
    Parameters:
        global_activity (float): Fraction of neurons firing in the liquid.
        weight (float): Current weight of the synapse.
        e_t (float): Update factor e(t).
        k (int): Gain parameter for scaling activity adjustment.
    Returns:
        float: Activity adjustment factor.
    """
    a_t = 2 ** (k * (global_activity - 0.5))  # Transform global activity
    if weight * e_t >= 0:
        return 1 / a_t
    else:
        return a_t

# Weight Update Implementation
def update_weights(weights_, state_vectors, class_labels, max_separation, mu_w, mw, learning_rate=0.001, first_id: int = 0):
    """
    Update synaptic weights using SDSM.
    Parameters:
        weights_ (np.ndarray): Current weights of the liquid.
        state_vectors (np.ndarray): Matrix of state vectors (rows = samples, cols = neurons).
        class_labels (np.ndarray): Array of class labels corresponding to state vectors.
        max_separation (float): Approximate maximum separation for the problem.
        global_activity (float): Fraction of neurons firing in the liquid.
        mu_w (float): Expected value of the magnitude of synaptic weights in the initial liquid.
        mw (float): Estimate of the maximum of the random values drawn from the initialization distribution.
        learning_rate (float): Learning rate for the weight update.
    Returns:
        np.ndarray: Updated weights.
    """
    # Reduce to random samples per class
    # reduced_vectors, reduced_labels = select_random_samples(state_vectors, class_labels)
    # Compute separation metrics and neuron properties
    cd = separation_metric(state_vectors, class_labels)
    neuron_activity, neuron_variances = calculate_neuron_activity_and_variances(state_vectors, class_labels)
    # Update weights
    weights = np.zeros((64, 64))
    for s, t, w in zip(weights_["source"], weights_["target"], weights_["weight"]):
        weights[s - first_id, t - first_id] = w
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            e_t = compute_e(weights[i, j], cd, max_separation, neuron_activity[i], neuron_variances[i], mu_w, mw)
            f_t = compute_f(neuron_activity[i], weights[i, j], e_t)
            delta_w = e_t * learning_rate * f_t
            weights[i, j] += np.sign(weights[i, j]) * (abs(weights[i, j]) + delta_w)
    new_weights = []
    for s, t, in zip(weights_["source"], weights_["target"]):
        new_weights.append(weights[s - first_id, t - first_id])
    return cd, new_weights

def binary_state(spikes: dict, first_id: int, n_neurons: int = 64, sim_time: int = 110., window: int = 50):
    spike_times = spikes["events"]['times']
    spike_senders = spikes["events"]["senders"]
    slice_start = sim_time - window
    slice_end = sim_time

    # Binary state
    b_state = np.zeros(n_neurons)
    for neuron in range(first_id, n_neurons + first_id):
        spike_in_slice = np.logical_and(spike_times>=slice_start, spike_times<=slice_end)
        b_state[neuron - first_id] = 1 if np.any(spike_senders[spike_in_slice] == neuron) else 0
    return b_state


class Norton_LSM:
    """
    Implementation of a Liquid State Machine (LSM) as described in Norton and Ventura's paper.

    This class simulates a Liquid State Machine using NEST, including configurable input channels, neuron parameters,
    synaptic connections, and noise. The LSM processes spike inputs and extracts neuronal states.

    Attributes:
        neuron_model (str): The NEST model for neurons (default is 'iaf_psc_delta').
        seed (int): Seed for random number generation to ensure reproducibility.
        channels (int): Number of input channels.
        neuron_params (dict): Parameters for the neuron model, including capacitance, time constants, thresholds, etc.
        conn_prob (float): Probability of connecting any two neurons.
        noiseI (float): Standard deviation of background noise input.
        weight_mean (float): Mean value of synaptic weights.
        weight_std (float): Standard deviation of synaptic weights.
        delay_mean (float): Mean synaptic delay.
        delay_std (float): Standard deviation of synaptic delay.
        synapses (None or list): Stores synaptic weights (initialized during the build phase).
        inputC (None or list): Input spike generators.
        monitor_spikes (None or list): Spike recorders to monitor neuron activity.
        neurons (None or list): Liquid neurons.
        noiseG (None or list): Noise generators for background activity.
    """

    neuron_model = 'iaf_psc_delta'

    def __init__(self, channels, seed):
        """
        Initialize the Liquid State Machine.
        Args:
            channels (int): Number of input channels.
            seed (int): Seed for random number generation.
        """
        self.seed = seed
        self.channels = channels
        self.neuron_params = {
            "C_m": 3e-08,  # Membrane capacitance
            "tau_m": 0.03,  # Membrane time constant
            "V_th": -0.045,  # Threshold potential
            "V_reset": -0.06,  # Reset potential
            "E_L": -0.06  # Resting potential
        }
        self.conn_prob = 0.3
        self.noiseI = 5e-8
        self.weight_mean = 2e-8
        self.weight_std = 4e-8
        self.delay_mean = 0.01
        self.delay_std = 0.1
        self.synapses = None
        self.inputC = None
        self.monitor_spikes = None
        self.neurons = None
        self.noiseG = None

    def setup(self):
        """
        Reset and prepare the NEST environment for the LSM simulation.
        Initializes input spike generators, neurons, noise sources, and spike monitors.
        """
        ResetKernel()
        SetKernelStatus({"rng_seed": self.seed, "resolution": 0.01})
        self.inputC = None
        self.neurons = None
        self.noiseG = None
        self.monitor_spikes = None
        CopyModel("static_synapse", "input_to_liquid_synapse")
        CopyModel("static_synapse", "liquid_synapse")
        CopyModel(existing='static_synapse', new='Monitor')

    def build(self):
        """
        Build the Liquid State Machine by creating neurons, input sources, and synaptic connections.
        Creates the following:
        - Input spike generators (self.inputC).
        - Liquid neurons (self.neurons).
        - Noise generators for background activity (self.noiseG).
        - Spike monitors to record neuron activity (self.monitor_spikes).
        Connects:
        - Inputs to Liquid neurons.
        - Liquid neurons to themselves.
        - Noise generators to Liquid neurons.
        - Liquid neurons to spike monitors.
        """
        # Layers
        self.inputC = Create(model="spike_generator", n=self.channels)
        self.neurons = Create(model=self.neuron_model, n=64, params=self.neuron_params)
        self.noiseG = Create(model="noise_generator", params={"mean": 0.0, "std": self.noiseI})
        self.monitor_spikes = Create(model="spike_recorder")
        # Connections
        Connect(
            self.inputC, self.neurons,
            conn_spec={"rule": "pairwise_bernoulli", "p": self.conn_prob},
            syn_spec={
                "synapse_model": "input_to_liquid_synapse",
                "weight": normal(self.weight_mean, self.weight_std),
                "delay": redraw(normal(self.delay_mean, self.delay_std), 0.01, 10)
            }
        )
        Connect(
            self.neurons, self.neurons,
            conn_spec={"rule": "pairwise_bernoulli", "p": self.conn_prob},
            syn_spec={
                "synapse_model": "liquid_synapse",
                "weight": normal(self.weight_mean, self.weight_std),
                "delay": redraw(normal(self.delay_mean, self.delay_std), 0.01, 10)
            },
            return_synapsecollection=True
        )
        weights = nest.GetConnections(synapse_model="liquid_synapse")
        if self.synapses is None:  # There is no previous synaptic weight connections
            self.synapses = weights.get(['weight', 'source', 'target'])
        else:  # Update the synaptic weights
            weights.set(weight=self.synapses['weight'])
        Connect(self.noiseG, self.neurons, syn_spec={"weight": 1})
        Connect(self.neurons, self.monitor_spikes, syn_spec={'synapse_model': 'Monitor'})

    def do_simulation(self, instances, time):
        """
        Run the simulation for a set of input instances and record Liquid neuron states.
        Args:
            instances (list): A list of input spike trains, where each element corresponds to an input channel.
            time (float): Duration of the simulation in milliseconds.
        Returns:
            list: A list of dictionaries containing spike data for each input instance.
            - Key 'events' contains the neuron IDs ('senders') and their spike times ('times').
        """
        self.setup()
        self.build()
        states = []
        biological_time = 0.
        for i, spike_train in enumerate(instances):
            # Assign spike trains to input generators
            SetStatus(self.inputC, [{"spike_times": spike_times + biological_time} for spike_times in spike_train])
            # Simulate
            Simulate(time)
            # Extract spike data
            spikes = self.monitor_spikes.get()
            spikes["events"]["times"] -= spikes["origin"]
            states.append(spikes)
            # Reset neuron membrane potentials and prepare for next instance
            self.neurons.V_m = -70.
            biological_time = GetKernelStatus("biological_time")
            self.monitor_spikes.n_events = 0
            self.monitor_spikes.origin = biological_time
            self.noiseG.origin = biological_time
        return states

def Norton_Methodology(classes, data_, channels, seed):
    # Variables
    sep = []
    acc = []
    sep_star = calculate_sep_star(classes, 64)
    # Define initial LSM (With initial weights)
    lsm = Norton_LSM(channels, seed)
    lsm.setup()
    lsm.build()
    initialSynapses = deepcopy(lsm.synapses)
    mu_w = mean(initialSynapses["weight"])
    mw = max(initialSynapses["weight"])
    first_id = initialSynapses["source"][0]
    for i in range(500):
        data = select_random_samples(data_["Spikes"], data_["Labels"])
        spike_states = lsm.do_simulation(data[0], 120.)
        bStates = np.array([binary_state(s, first_id, window=10) for s in spike_states])
        separation, new_w = update_weights(lsm.synapses, bStates, data[1], sep_star, mu_w, mw, first_id=first_id)
        sep.append(separation)
        lsm.synapses["weight"] = new_w
        print(f"M{seed-1} - Iteration {i+1}/500: Separation {separation}")
        if i == 0 or i == 499:
            classifier = Perceptron()
            spike_states = lsm.do_simulation(data_["Spikes"], 120.)
            bStates = np.array([binary_state(s, first_id, window=10) for s in spike_states])
            classifier.fit(bStates[:data_["Point"]], data_["Labels"][:data_["Point"]])
            acc_r = classifier.score(bStates["Spikes"][data_["Point"]:], data_["Labels"][data_["Point"]:])
            print("Accuracy: ", acc_r)
            acc.append(acc_r)
    return sep, acc

if __name__ == "__main__":
    import pickle
    from lib.auxiliary import split_data
    task = 'FR5'
    with open(f"../data/{task}.data", "rb") as file:
        data_all = pickle.load(file)
    data, _ = split_data(classes=data_all["Classes"], data=data_all, p_validation=0.7, p_train=0.75)
    all_results = []
    for i in range(30):
        results = Norton_Methodology(data_all["Classes"], data, data_all["Channels"], i + 1)
        all_results.append(results)

        with open(f"Results/Norton_{task}-{i+1}.data", "wb") as f:
            pickle.dump({"Results": all_results}, file=f)

    # sep_star = calculate_sep_star(5, 64)
    # lsm = Norton_LSM(4, 1)
    # lsm.setup()
    # lsm.build()
    # initialSynapses = deepcopy(lsm.synapses)
    # mu_w = mean(initialSynapses["weight"])
    # mw = max(initialSynapses["weight"])
    # first_id = initialSynapses["source"][0]
    # data = select_random_samples(data["Spikes"], data["Labels"])
    # spike_states = lsm.do_simulation(data[0], 150.)
    # bStates = np.array([binary_state(s, 5, window=10) for s in spike_states])
    # separation, new_w = update_weights(lsm.synapses, bStates, data[1], sep_star, mu_w, mw, first_id=first_id)
    # lsm.synapses["weight"] = new_w
    # print(f"Separation {separation}")










