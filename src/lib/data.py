# General imports
from numpy import ndarray, floor as npFloor, array as npArray
from numpy.random import rand as npRand, randint as npRandInt, normal as npNormal, uniform as npUniform
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw, ImageFont

# NEST imports
from nest.lib.hl_api_nodes import Create
from nest.lib.hl_api_simulation import Simulate, ResetKernel, SetKernelStatus
from nest.lib.hl_api_connections import Connect
# Tonic imports
from tonic import transforms
from tonic.transforms import ToFrame, ToTimesurface, ToVoxelGrid, Denoise, MergePolarities, CenterCrop, UniformNoise, CropTime, Downsample
from tonic.utils import plot_animation
from tonic.datasets import NMNIST

def poisson_process(rate: float, duration: float, bin_size: float, jitter: int = 10) -> ndarray:
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
    # Adjustment: Since low frequency can, in some cases, create the scenario where no spikes are produced, we fixed the
    #               lower bound of the generator at the given rate. In exchange, the higher bound at twice the jitter.
    #               This will take effect if the given frequency * the duration is lower or equal to 1.
    if rate * duration <= 1:
        jitter_rate = npRandInt(low= int(rate), high= int(rate) + jitter * 2)
    else:
        jitter_rate = npRandInt(low=int(rate) - jitter, high=int(rate) + jitter)
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

def nest_poisson(channels: int, rates: list, tmax: float) -> dict:
    """
    Method to produce poisson spike trains based on NEST's poisson_generator.
    :param channels: Number of channels.
    :param rates: List of rates for each channel.
    :param tmax: Duration of the signal.
    :return: Dictionary with the spike events.
    """
    # Reset kernel
    ResetKernel()
    # Do random seed
    SetKernelStatus({"rng_seed": npRandInt(1, 100000)})
    # Build Poisson generator
    poissonGen = Create(model="poisson_generator", n=channels, params={"rate":rates})
    # Build monitor to record poisson generator
    monitor = Create(model="spike_recorder")
    # Connect
    Connect(poissonGen, monitor)
    # Do simulation
    Simulate(tmax)
    # Return events
    return monitor.get()

def frequency_recognition_data(size: int, classes: int, channels: int, rates: list, tmax: float, name: str,
                               jitter: float = 5., labels_names: list | None = None) -> dict:
    """
    Datasets for the Frequency recognition problem.
    :param size: Number of instances for each class.
    :param classes: Number of classes.
    :param channels: Number of channels.
    :param rates: Rates for each channel.
    :param tmax: Duration of signals.
    :param name: Name of the dataset.
    :param jitter: Variation of the original rate (Hz).
    :param labels_names: Name of the classes.
    :return: Dictionary for Frequency recognition problem.
    """
    # Initialize spike list
    spikes = []
    for _ in range(size):
        # Create instance of given rate
        for class_i in range(classes):
            instance = []
            # Jitter rates
            rate = [npRandInt(r - jitter, r + jitter) for r in rates[class_i]]
            poissonNeurons = nest_poisson(channels, rate, tmax)["events"]
            # Obtain activity by channel
            for c in range(channels):
                activity = [t for n, t in zip(poissonNeurons["senders"], poissonNeurons["times"]) if n == c + 1 ]
                instance.append(npArray(activity))
            spikes.append(instance)
    labels = npArray([l for _ in range(size) for l in range(classes)])
    # Build dictionary of data
    data: dict = {"Spikes": spikes, "Labels": labels, "Channels": channels, "Classes": classes, "Tmax": tmax,
                  "Labels Names": labels_names}
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
                             std_jitter: float = 5., labels_names: list | None = None) -> dict:
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
    :param labels_names: Name of the classes.
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
                  "Channels": channels, "Classes": classes, "Tmax": tmax, "Labels Names": labels_names}
    # Save data dile
    with open(f"../Experiments/Datasets/{name}.data", "wb") as file:
        pickle.dump(data, file)
    return data


def nmnist(name: str, root: str, plot: bool =True):
    # noinspection PyTypeChecker
    transform = transforms.Compose([MergePolarities(),
                                    UniformNoise(sensor_size=(34, 34, 1), n=800),
                                    CropTime(min=5000, max=200000),
                                    Downsample(spatial_factor=0.5)])
    # noinspection PyTypeChecker
    to_frame = transforms.ToFrame(sensor_size=(17, 17, 1), time_window=10000)
    nm1 = NMNIST(root, train=True, transform=transform)
    nm2 = NMNIST(root, train=False, transform=transform)
    if plot:
        gifs = []
        for i in range(10):
            j = nm1.targets.index(i)
            events, label = nm1[j]
            ani = plot_animation(to_frame(events))

            # Save individual animations as temporary GIFs
            ani_writer = animation.PillowWriter()
            temp_gif = f"NMNIST_{label}.gif"
            ani.save(temp_gif, writer=ani_writer)
            gifs.append(temp_gif)

        # Combine GIFs into a single grid animation
        combined_gif = "NMNIST.gif"
        combine_gifs(gifs, combined_gif, grid_size=(3, 3))  # Adjust grid_size if needed
        print(f"Combined GIF saved as {combined_gif}")

    spikes = []
    labels = []
    for data in [*nm1, *nm2]:
        events, l = data
        labels.append(l)
        channels = [[] for _ in range(17 * 17)]
        for e in events:
            i = (17 * e[0]) + e[1]
            channels[i].append(round(e[2] / 1000))
        spikes.append([npArray(c) for c in channels])

    dataset = {"Spikes": spikes, "Labels": npArray(labels), "Channels": 17 * 17, "Classes": 10,
               "Tmax": 200000 / 1000, "Labels Names": [f"C{str(c)}" for c in range(10)]}
    # Save data file
    with open(f"../Experiments/Datasets/{name}.data", "wb") as file:
        pickle.dump(dataset, file)
    return dataset

def combine_gifs(gif_paths, output_path, grid_size=(3, 3)):
    frames = []
    images = [Image.open(gif) for gif in gif_paths]

    # Assuming all GIFs have the same number of frames
    frame_count = images[0].n_frames

    for frame_idx in range(frame_count):
        grid_images = []
        for img in images:
            img.seek(frame_idx)
            grid_images.append(img.copy())

        # Arrange images in a grid
        grid = create_grid(grid_images, grid_size)
        frames.append(grid)

    # Save combined frames as a single GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=100)


def create_grid(images, grid_size):
    """Arrange images in a grid and return a single combined image."""
    rows, cols = grid_size
    img_width, img_height = images[0].size
    grid_img = Image.new('L', (cols * img_width, rows * img_height))

    for idx, img in enumerate(images):
        x = (idx % cols) * img_width
        y = (idx // cols) * img_height
        grid_img.paste(img, (x, y))

    return grid_img

if __name__ == '__main__':
    import numpy as np
    with open("../Experiments/Datasets/PR12.data", "rb") as file:
        example = pickle.load(file)

    nmn = nmnist("Nmnist","../Experiments/Datasets/N_MNIST", plot=False)
    # from visualization import plot_input_classes, plot_PR_visualization

    # h = 100 # High Freq
    # l = 10 # Low Freq

    # # ---------------------------------------- Frequency Recognition
    # # --------------------------- Two classes: FR2
    # # Small FR2 Jitter
    # data = frequency_recognition_data(size=50, classes=2, channels=3, rates=[[h, l, l],
    #                                                                          [l, h, l]],
    #                                   tmax=100., name="FR2_small", labels_names= ["C1", "C2"])
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1]], name="FR2 - Small - Jitter")
    # # Middle FR2 Jitter
    # data = frequency_recognition_data(size=500, classes=2, channels=3, rates=[[h, l, l],
    #                                                                           [l, h, l]],
    #                                   tmax=100., name="FR2_middle", labels_names= ["C1", "C2"])
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1]], name="FR2 - Middle - Jitter")
    # # Large FR2 Jitter
    # data = frequency_recognition_data(size=1000, classes=2, channels=3, rates=[[h, l, l],
    #                                                                            [l, h, l]],
    #                                   tmax=100., name="FR2", labels_names= ["C1", "C2"])
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1]], name="FR2")
    # # --------------------------- Five classes: FR5
    # # Small FR5 Jitter
    # data = frequency_recognition_data(size=50, classes=5, channels=4, rates=[[h, l, l, l],
    #                                                                          [l, h, l, l],
    #                                                                          [h, h, l, l],
    #                                                                          [l, l, h, l],
    #                                                                          [h, l, h, l]],
    #                                   tmax=100., name="FR5_small", labels_names= ["C1", "C2", "C3", "C4", "C5"])
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1],
    #                             data["Spikes"][2],
    #                             data["Spikes"][3],
    #                             data["Spikes"][4]], name="FR5 - Small - Jitter")
    #
    # # Middle FR5 Jitter
    # data = frequency_recognition_data(size=500, classes=5, channels=4, rates=[[h, l, l, l],
    #                                                                           [l, h, l, l],
    #                                                                           [h, h, l, l],
    #                                                                           [l, l, h, l],
    #                                                                           [h, l, h, l]],
    #                                   tmax=100., name="FR5_middle", labels_names= ["C1", "C2", "C3", "C4", "C5"])
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1],
    #                             data["Spikes"][2],
    #                             data["Spikes"][3],
    #                             data["Spikes"][4]], name="FR5 - Middle - Jitter")
    #
    # # Large FR5 Jitter
    # data = frequency_recognition_data(size=1000, classes=5, channels=4, rates=[[h, l, l, l],
    #                                                                            [l, h, l, l],
    #                                                                            [h, h, l, l],
    #                                                                            [l, l, h, l],
    #                                                                            [h, l, h, l]],
    #                                   tmax=100., name="FR5", labels_names= ["C1", "C2", "C3", "C4", "C5"])
    # plot_input_classes(classes=[data["Spikes"][0],
    #                             data["Spikes"][1],
    #                             data["Spikes"][2],
    #                             data["Spikes"][3],
    #                             data["Spikes"][4]], name="FR5")
    #
    # # ---------------------------------------- Pattern Recognition
    # # --------------------------- Four classes: PR4
    # # Small PR4 Jitter
    # data = pattern_recognition_data(size=50, classes=4, name="PR4_small", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4"])
    # plot_input_classes(data["Patterns"], name="PR4 - Small - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR4 - Small - Jitter")
    #
    # # Middle PR4 Jitter
    # data = pattern_recognition_data(size=500, classes=4, name="PR4_middle", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4"])
    # plot_input_classes(data["Patterns"], name="PR4 - Middle - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR4 - Middle - Jitter")
    #
    # # Middle PR4 Jitter
    # data = pattern_recognition_data(size=1000, classes=4, name="PR4", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4"])
    # plot_input_classes(data["Patterns"], name="PR4")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR4")
    #
    # # --------------------------- Four classes: PR8
    # # Small PR8 Jitter
    # data = pattern_recognition_data(size=50, classes=8, name="PR8_small", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
    # plot_input_classes(data["Patterns"], name="PR8 - Small - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR8 - Small - Jitter")
    #
    # # Middle PR8 Jitter
    # data = pattern_recognition_data(size=500, classes=8, name="PR8_middle", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
    # plot_input_classes(data["Patterns"], name="PR8 - Middle - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR8 - Middle - Jitter")
    #
    # # Middle PR8 Jitter
    # data = pattern_recognition_data(size=1000, classes=8, name="PR8", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
    # plot_input_classes(data["Patterns"], name="PR8")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR8")
    #
    # # --------------------------- Four classes: PR12
    # # Small PR12 Jitter
    # data = pattern_recognition_data(size=50, classes=12, name="PR12_small", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"])
    # plot_input_classes(data["Patterns"], name="PR12 - Small - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR12 - Small - Jitter")
    #
    # # Middle PR12 Jitter
    # data = pattern_recognition_data(size=500, classes=12, name="PR12_middle", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"])
    # plot_input_classes(data["Patterns"], name="PR12 - Middle - Jitter")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR12 - Middle - Jitter")
    #
    # # Middle PR12 Jitter
    # data = pattern_recognition_data(size=1000, classes=12, name="PR12", tmax=100., mean_distance=10.,
    #                                 std_distance=20., mean_jitter=0., std_jitter=5.,
    #                                 labels_names= ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"])
    # plot_input_classes(data["Patterns"], name="PR12")
    # plot_PR_visualization(data["Patterns"], data["Spikes"], name="PR12")





