# General imports
from time import time
from numpy import ndarray, array as npArray, zeros as npZeros
from numpy.random import permutation as npPermutation

class ExecutionTime:
    def __init__(self, run_time: float = 0.) -> None:
        self.start: float = time()  # Store the time when the instance is created
        self.run_time: float = run_time  # Cumulative runtime (optional starting value)

    def update(self) -> None:
        """Updates the total run_time by calculating the elapsed time."""
        current_time = time()
        self.run_time += current_time - self.start
        self.start = current_time  # Reset start for the next update

    def report_elapsed(self) -> float:
        """Reports the time elapsed since the last update."""
        return time() - self.start  # Directly return the elapsed time since last start

    def report_run_time(self) -> float:
        """Returns the total run_time so far."""
        return self.run_time

def spike_count(data: dict, n_neurons: int, window: float = 0., first_id: int = 4) -> ndarray:
    """
    Spike count. It counts the spikes in a window time.
    :param data: Recorded spike activity.
    :param n_neurons: Number of neurons to count spikes.
    :param window: Time window to count spikes. It filters early spikes.
    :param first_id: Id of the first excitatory neuron to count spikes. The number of input channels affects this.
    :return: Array of spike counts (Spike count per neuron).
    """
    # Obtain spike activity from data
    neurons: ndarray[int] = data["events"]["senders"] - first_id
    times: ndarray[int] = data["events"]["times"]
    # Initialize count
    count: ndarray = npZeros(n_neurons)
    # Perform count depending on the window
    for neuron, t in zip(neurons, times):
        if t > window:
            count[neuron] += 1
    return count

def spike_rate(data: dict, sim_time: float, n_neurons: int, window: float = 0., first_id: int = 4) -> ndarray:
    """
        Spike rate.
        :param data: Recorded spike activity.
        :param n_neurons: Number of neurons to count spikes.
        :param sim_time: Time of simulation
        :param window: Time window to count spikes. It filters early spikes.
        :param first_id: Id of the first excitatory neuron to count spikes. The number of input channels affects this.
        :return: Array of spike rates (Spike rate per neuron).
        """
    return spike_count(data, n_neurons, window, first_id) / (sim_time - window)

class DataLoader:
    """ Class to load data"""
    def __init__(self, spike_data: list, labels: ndarray) -> None:
        self.spike_data = spike_data
        self.labels = labels
        self.n = labels.shape[0]  # Dimension of data
        # Initialize variable for permutation
        self.permutation = None
        # Do initial permutation. This process is made in a separate method to be able to redo it later on.
        self.permute()

    def permute(self) -> None:
        """ Generate the random permutation"""
        self.permutation = npPermutation(range(self.n))

    def permuted_samples(self) -> tuple[list, ndarray]:
        """ Return the permutation samples"""
        return [self.spike_data[i] for i in self.permutation], npArray([self.labels[i] for i in self.permutation])

    def __getitem__(self, item) -> tuple[ndarray, int | float | str]:
        """ Iterate over samples according to the permutation order"""
        return self.spike_data[self.permutation[item]], self.labels[self.permutation[item]]

def split_data(classes: int, data: dict, p_train: float, p_validation: float) -> tuple[dict, dict]:
    """
    Method to split data in two sets: For evolutionary process, and for validation. This is tailored for data produced
        for the frequency & pattern recognition task made in data.py
    Both the spike data & labels are returned in a random permutation. TODO: Make it as to return the DataLoader directly
    :param classes: Number of classes for the task.
    :param data: Dictionary of data (spikes and labels) to be split.
    :param p_train: Percentages of data to be used in evolutionary process (Train set). Not complementary with p_validation. It may be 1.0, from which the test set becomes empty
    :param p_validation: Percentage of data to be used in validation process.
    :return: Dictionaries for evolutionary process (train and test) and validation process.
    """
    # Data is arrayed sequentially: Class1, Class2, ..., Class1, Class2, ...
    # Then, the point of split must be a multiple of the number of classes.
    # I think this is guaranteed since the generation of the database, and since all classes are balanced.
    len_data = data["Labels"].shape[0]
    # Selection of evolutionary train process
    sel_evolution = int((1 - p_validation) * len_data)
    # Get a random permutation of data from DataLoader
    data_loader = DataLoader(spike_data=data["Spikes"][sel_evolution:], labels=data["Labels"][sel_evolution:])
    data_spikes, data_labels = data_loader.permuted_samples()
    # -------------------- Data validation
    data_validation: dict = {"Spikes": data_spikes, "Labels": data_labels}
    # -------------------- Data evolution
    # Get a random permutation of data from DataLoader
    data_loader = DataLoader(spike_data=data["Spikes"][:sel_evolution], labels=data["Labels"][:sel_evolution])
    data_evolution_spikes, data_evolution_labels = data_loader.permuted_samples()
    # Data for evolution needs further split in training and testing subsets
    sel_training = int(p_train * sel_evolution)
    # This used to ensure to have a perfect balanced training and testing sets. Since instances are now random, this is
    # no longer achievable. Nevertheless, we are keeping it for latter improvement.
    if sel_training % classes:
        sel_training += classes - sel_training % classes
    data_evolution: dict = {"Spikes": data_evolution_spikes, "Labels": data_evolution_labels, "Point": sel_training}
    return data_evolution, data_validation

if __name__ == "__main__":
    import pickle
    with open("../Experiments/Datasets/FR5_small_jitter.data", "rb") as file:
        data_in = pickle.load(file)

