# General imports
from numpy import ndarray, floor as npFloor, array as npArray
from numpy.random import rand as npRand, randint as npRandInt, normal as npNormal, uniform as npUniform
import pickle

def poisson_process(rate: float, duration: float, bin_size: float, jitter: int = 50) -> ndarray:
    """
    Adapted from Bartosz Telenczuk, 2017. Python in neuroscience - tutorials. Zenodo. doi:10.5281/zenodo.345422.
    :param rate: Poisson rate (Hz)
    :param duration: Duration of spike train (s)
    :param bin_size: Bin (block) size (s)
    :param jitter: Posible variation (-+) of the original rate (Hz)
    :return: Poisson spike times (ms)
    """
    # Length of spike train (s) to bins (sections, s). This changes it from seconds to blocks of ms
    n_bins: int = npFloor(duration / bin_size).astype(int)
    jitter_rate = npRandInt(low= int(rate) - jitter, high= int(rate) + jitter)
    # Probability based on rate and the bin dimension
    prob_of_spike: float = jitter_rate * bin_size
    # Generate random numbers and compare them to probability
    spikes: ndarray = npRand(n_bins) < prob_of_spike
    # Save the time (ms) when a spike occurs
    time_spikes: ndarray = npArray([float(t) for t, spiked in enumerate(spikes) if spiked])
    return time_spikes + 1  # +1 to ensure no spike time is set to 0

def poisson_gen(channels: int, rate: int | list | tuple, tmax: float = 1.0, bin_size: float = 0.001) -> list:
    """
    Generates as many poisson spike trains as channels per sample.
    :param channels: Number of channels (Inputs)
    :param rate: Poisson rate (Hz)
    :param tmax: Duration of spike train (s)
    :param bin_size: Bin (block) size (s)
    :return: Channels of Poisson spike times (ms)
    """
    while True:
        channels_spikes: list = []
        if isinstance(rate, int):
            for channel in range(channels):
                channels_spikes.append(poisson_process(rate, tmax, bin_size))
        else:
            for r in rate:
                channels_spikes.append(poisson_process(r, tmax, bin_size))
        yield channels_spikes

def frequency_recognition_data(size: int, classes: int, channels: int, rates: list, tmax: float, bin_size: float, name: str) -> dict:
    """
    Datasets for the Frequency recognition problem.
    :param size: Size of the dataset.
    :param classes: Number of classes.
    :param channels: Number of channels.
    :param rates: Rates for each channel.
    :param tmax: Duration of the signal.
    :param bin_size: Size of the bin.
    :param name: File name to save the data.
    :return: Dataset for Pattern recognition problem.
    """
    # Prepare poisson generators
    ds = []
    for i in range(classes):
        _d = poisson_gen(channels, rates[i], tmax, bin_size)
        ds.append(_d)
    # Get spike trains & labels
    spikes = [next(d) for _ in range(size) for d in ds]
    labels = npArray([l for _ in range(size) for l in range(classes)])
    # Build dictionary
    data: dict = {"Spikes": spikes, "Labels": labels, "Channels": channels, "Classes": classes, "Tmax": tmax * 1000.}
    # Save dataset
    with open(f"../Experiments/Datasets/{name}.data", "wb") as file:
        pickle.dump(data, file)
    return data

def pattern_gen(tmax: float, mean_distance: float, std_distance: float) -> ndarray:
    """
    Generates a spike pattern based on the distance between each spike.
    :param tmax: Duration of the spike train
    :param mean_distance: Mean distances between spikes
    :param std_distance: Standar deviation distance between spikes
    :return: Spike train pattern
    """
    # Set empty patter.
    pattern = []
    # Set initial spike. It must fall in the first mean_distance value.
    current_spike: int = int(npUniform(1., mean_distance))
    # Generate sequential spikes until the maximum time has been reached.
    while current_spike < tmax:
        # Add spike.
        pattern.append(current_spike)
        # Generate next spike, ensuring a distance drawn from N(mean_distance, std_distance)
        current_spike += abs(int(npNormal(mean_distance, std_distance)))
    return npArray(pattern)

def patter_jittering(pattern: ndarray, mean_jitter: float, std_jitter: float, tmax: float) -> ndarray:
    """
    Generates a random modification of a spike pattern based on jittering.
    :param pattern: Pattern to modify.
    :param mean_jitter: Mean jittering distance
    :param std_jitter: Standar jittering distance
    :param tmax: Duration of the spike train
    :return:
    """
    # Initialize the jitter pattern
    jitter_pattern = []
    # Iterate over each spike to jitter it
    for i, spike_time in enumerate(pattern):
        # Assure that there is spike time <= 0
        current_time = pattern[i]
        jittering = int(npNormal(mean_jitter, std_jitter))
        # Assure that the spike time is greater than 0. If not, don't jitter the spike.
        if 0. < current_time + jittering < tmax:
            # Jitter the spike time
            jitter_pattern.append(current_time + jittering)
        else:
            jitter_pattern.append(current_time)
    # Sort pattern to have increasing spike times
    jitter_pattern.sort()
    return npArray(jitter_pattern)


def pattern_recognition_data(size: int, classes: int, name: str, channels: int = 8, tmax: float = 100.,
                             mean_distance: float = 10., std_distance: float = 20, mean_jitter: float = 0.,
                             std_jitter: float = 5.) -> dict:
    """
    Datasets for the Pattern recognition problem.
    :param size: Size of the dataset.
    :param classes: Number of classes.
    :param name: File name to save the data.
    :param channels: Number of channels.
    :param tmax: Duration of the signal.
    :param mean_distance: Mean distances between spikes for pattern creation.
    :param std_distance: Standard deviation distance between spikes for pattern creation.
    :param mean_jitter: Mean jittering distance.
    :param std_jitter: Standard jittering distance.
    :return: Dataset for Pattern recognition problem.
    """
    # Generate classes
    classes_patterns = []
    for _ in range(classes):
        # Generate random pattern for each input
        input_pattern = []
        for _ in range(channels):
            input_pattern.append(pattern_gen(tmax, mean_distance, std_distance))
        classes_patterns.append(input_pattern)
    # Generate instances
    spikes_instances = []
    for _ in range(size):
        for input_pattern in classes_patterns:
            channel_activity = []
            for pattern in input_pattern:
                channel_activity.append(patter_jittering(pattern, mean_jitter, std_jitter, tmax))
            spikes_instances.append(channel_activity)
    # Generate Labels
    labels = npArray([l for _ in range(size) for l in range(classes)])
    # Generate Data
    data: dict = {"Spikes": spikes_instances, "Labels": labels, "Patterns": classes_patterns,
                  "Channels": channels, "Classes": classes, "Tmax": tmax}
    # Save data dile
    with open(f"../Experiments/Datasets/{name}.data", "wb") as file:
        pickle.dump(data, file)
    return data

if __name__ == '__main__':
    pass

    # # ---------------------------------------- Frequency Recognition
    # # --------------------------- Two classes: FR2
    # # Small FR2 Jitter
    # data = frequency_recognition_data(size=50, classes=2, channels=3, rates=[[500, 100, 100],
    #                                                                          [100, 500, 100]],
    #                                   tmax=0.1, bin_size=0.001, name="FR2_small_jitter")
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1]], name="FR2 - Small - Jitter")
    #
    # # Middle FR2 Jitter
    # data = frequency_recognition_data(size=500, classes=2, channels=3, rates=[[500, 100, 100],
    #                                                                           [100, 500, 100]],
    #                                   tmax=0.1, bin_size=0.001, name="FR2_middle_jitter")
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1]], name="FR2 - Middle - Jitter")
    #
    # # Large FR5 Jitter
    # data = frequency_recognition_data(size=1000, classes=2, channels=3, rates=[[500, 100, 100],
    #                                                                            [100, 500, 100]],
    #                                   tmax=0.1, bin_size=0.001, name="FR2_jitter")
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1]], name="FR2")
    #
    # # --------------------------- Five classes: FR5
    # # Small FR5 Jitter
    # data = frequency_recognition_data(size=50, classes=5, channels=4, rates=[[500, 100, 100, 100],
    #                                                                          [100, 500, 100, 100],
    #                                                                          [500, 500, 100, 100],
    #                                                                          [100, 100, 500, 100],
    #                                                                          [500, 100, 500, 100]],
    #                                   tmax=0.1, bin_size=0.001, name="FR5_small_jitter")
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1],
    #                             data["Spikes"][2],
    #                             data["Spikes"][3],
    #                             data["Spikes"][4]], name="FR5 - Small - Jitter")
    #
    # # Middle FR5 Jitter
    # data = frequency_recognition_data(size=500, classes=5, channels=4, rates=[[500, 100, 100, 100],
    #                                                                           [100, 500, 100, 100],
    #                                                                           [500, 500, 100, 100],
    #                                                                           [100, 100, 500, 100],
    #                                                                           [500, 100, 500, 100]],
    #                                   tmax=0.1, bin_size=0.001, name="FR5_middle_jitter")
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1],
    #                             data["Spikes"][2],
    #                             data["Spikes"][3],
    #                             data["Spikes"][4]], name="FR5 - Middle - Jitter")
    #
    # # Large FR5 Jitter
    # data = frequency_recognition_data(size=1000, classes=5, channels=4, rates=[[500, 100, 100, 100],
    #                                                                            [100, 500, 100, 100],
    #                                                                            [500, 500, 100, 100],
    #                                                                            [100, 100, 500, 100],
    #                                                                            [500, 100, 500, 100]],
    #                                   tmax=0.1, bin_size=0.001, name="FR5_jitter")
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1],
    #                             data["Spikes"][2],
    #                             data["Spikes"][3],
    #                             data["Spikes"][4]], name="FR5")

    # # ---------------------------------------- Pattern Recognition
    # # --------------------------- Four classes: PR4
    # # Small PR4 Jitter
    # data = pattern_recognition_data(size=50, classes=4, name="PR4_small", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR4 - Small - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR4 - Small - Jitter")
    #
    # # Middle PR4 Jitter
    # data = pattern_recognition_data(size=500, classes=4, name="PR4_middle", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR4 - Middle - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR4 - Middle - Jitter")
    #
    # # Middle PR4 Jitter
    # data = pattern_recognition_data(size=1000, classes=4, name="PR4", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR4")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR4")
    #
    # # --------------------------- Four classes: PR8
    # # Small PR8 Jitter
    # data = pattern_recognition_data(size=50, classes=8, name="PR8_small", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR8 - Small - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR8 - Small - Jitter")
    #
    # # Middle PR8 Jitter
    # data = pattern_recognition_data(size=500, classes=8, name="PR8_middle", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR8 - Middle - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR8 - Middle - Jitter")
    #
    # # Middle PR8 Jitter
    # data = pattern_recognition_data(size=1000, classes=8, name="PR8", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR8")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR8")
    #
    # # --------------------------- Four classes: PR12
    # # Small PR12 Jitter
    # data = pattern_recognition_data(size=50, classes=12, name="PR12_small", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR12 - Small - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR12 - Small - Jitter")
    #
    # # Middle PR12 Jitter
    # data = pattern_recognition_data(size=500, classes=12, name="PR12_middle", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR12 - Middle - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR12 - Middle - Jitter")
    #
    # # Middle PR12 Jitter
    # data = pattern_recognition_data(size=1000, classes=12, name="PR12", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.)
    # plot_input_classes(data["Patterns"], name="PR12")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR12")





