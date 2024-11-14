# General imports
import matplotlib.pyplot as plt
from numpy import ndarray, array as npArray, isnan, isinf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from seaborn import color_palette
# NEST imports
from nest.lib.hl_api_spatial import GetPosition
from nest.lib.hl_api_connections import GetConnections
from nest.lib.hl_api_types import SynapseCollection


def plot_vm_separate(data: dict, net, w_threshold: bool = False, name:str = "Membrane Potential per Neuron") -> None:
    """
    Plots each neuron's membrane potential on separate subplots for better clarity.
    :param data: Data from the simulation as given by NEST.
    :param net: LSM net.
    :param w_threshold: Option to plot threshold bar.
    :param name: Title of the plot.
    :return: None
    """
    # Get Threshold values.
    if w_threshold:
        thr = net.neuronsE.get("V_th") + net.neuronsI.get("V_th")
    # Define subplot grid dimensions
    cols = 3  # Number of columns in the subplot grid
    rows = (net.nT + cols - 1) // cols  # Calculate rows needed
    # Prepare the figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True, sharey=True)
    fig.suptitle(name, fontsize=16)
    # Flatten axes for easy iteration, in case rows * cols > n_neurons
    axes = axes.flatten()
    for i in range(net.nT):
        # Determine neuron type and assign color
        color = "green" if i < net.nE else "red"
        # Extract time and membrane potential data for neuron i
        sender: int = data["events"]["senders"][i::net.nT][-1]
        x_time: list[float] = data["events"]["times"][i::net.nT]
        y_vm: list[float] = data["events"]["V_m"][i::net.nT]
        # Plot in the appropriate subplot
        axes[i].plot(x_time, y_vm, color=color)
        if w_threshold:
            axes[i].axhline(y=thr[i], color='blue', linestyle='--', label="Threshold")
        axes[i].set_title(f"Neuron {sender} ({'Excitatory' if color == 'green' else 'Inhibitory'})")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Voltage (mV)")
    # Create a reference line for the legend outside the loop
    if w_threshold:
        line = plt.Line2D([0], [0], color='blue', linestyle='--', label="Threshold")
        fig.legend(handles=[line], loc="upper right")
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.show()

def plot_vm(data: dict, n_neurons: int, indexes: list[int]) -> None:
    """
    Plots the membrane voltage of an LSM simulation.
    :param data: Data from the simulation as given by NEST.
    :param n_neurons: Number of neurons to be plotted.
    :param indexes: List of indexes for excitatory neurons.
    :return: None
    """
    # Convert indexes to a set for O(1) lookup
    excitatory_set = set(indexes)
    # Prepare the plot
    plt.figure("Plot_Vm", figsize=(10, 6))
    plt.title("Membrane Potential")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    # Plot each neuron's membrane potential
    for i in range(n_neurons):
        # Determine neuron type and assign color
        color = "green" if i in excitatory_set else "red"
        # Extract time data (not assuming equal length for all neurons)
        x_time: list[float] = data["events"]["times"][i::n_neurons]
        # Extract the membrane potential for neuron i
        y_vm: list[float] = data["events"]["V_m"][i::n_neurons]
        # Plot the voltage trace
        plt.plot(x_time, y_vm, color=color)
    # Display the plot
    plt.show()

def plot_spikes(data: dict, n_neurons: int, first_id: int = 4, name: str = "Spikes") -> None:
    """
    Plots the spikes recover from the LSM network.
    :param data: Data to be plotted.
    :param n_neurons: Number of neurons to plot.
    :param first_id: ID of the first excitatory neuron to plot.
    :param name: Title of the plot.
    :return: None
    """
    # Retrieve information
    neurons: list[int] = data['events']['senders'] - first_id  # Exicitatory neurons do not start in 0.
    times: list[float] = data['events']['times']
    # Perform ploting
    plt.figure('Plot_Spikes', figsize=(10, 6))
    plt.title(name)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neurons')
    plt.plot(times, neurons, "|", color='black')
    plt.yticks(range(n_neurons))
    plt.tight_layout()
    plt.show()

def plot_input_data(sample: list, name: str = "Inputs") -> None:
    """
    Plots the input data (Spikes).
    :param sample: Spike information.
    :param name: Title of the plot.
    :return: None
    """
    # Get number of inputs.
    n_channels: int = len(sample)
    plt.figure('Plot_Inputs', figsize=(10, 6))
    plt.title(name)
    plt.xlabel('Time (ms)')
    # Iterate over channels of the sample to plot them.
    for channel, spike_times in enumerate(sample):
        _channel: list[int] = [channel for _ in range(len(spike_times))]
        plt.plot(spike_times, _channel, '|', color='black')
    plt.yticks(range(n_channels), [f'Channel {channel}' for channel in range(n_channels)])
    plt.tight_layout()
    plt.show()

def do_pca(data: ndarray, labels: list, dim: int = 2, plot: bool = True, name="PCA") -> ndarray:
    """
    Performs PCA on the data for 2D visualization.
    :param data: Data to be reduced.
    :param labels: Labels of classes.
    :param dim: Dimension of reduction.
    :param plot: Option to plot.
    :param name: Title of the plot.
    :return: Reduction of states.
    """
    # Do reduction
    x_reduce: ndarray = PCA(n_components=dim).fit_transform(data)
    # Do plot
    if plot:
        plt.figure('PCA')
        plt.title(name)
        n_classes = len(labels)
        cmap = color_palette("tab10", n_classes) if n_classes <= 10 else color_palette("hsv", n_classes)
        for i, label in enumerate(labels):
            plt.scatter(x_reduce[i::2, 0], x_reduce[i::2, 1], c=[cmap[i]], label=label)
        plt.legend()
        plt.show()
    return x_reduce

def do_tsne(data: ndarray, labels: list, dim: int = 2, plot: bool = True, name="T-SNE") -> ndarray:
    """
    Performs t-SNE on the data for 2D visualization.
    :param data: Data to be reduced.
    :param labels: Labels of classes.
    :param dim: Dimension of reduction.
    :param plot: Option to plot.
    :param name: Title of the plot.
    :return: Reduction of states.
    """
    # Check for NaNs or Infs
    if isnan(data).any() or isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values.")
    # Do reduction
    x_reduce: ndarray = TSNE(n_components=dim).fit_transform(data)
    # Do plot
    if plot:
        plt.figure('T-SNE')
        plt.title(name)
        n_classes = len(labels)
        cmap = color_palette("tab10", n_classes) if n_classes <= 10 else color_palette("hsv", n_classes)
        for i, label in enumerate(labels):
            plt.scatter(x_reduce[i::2, 0], x_reduce[i::2, 1], c=[cmap[i]], label=label)
        plt.legend()
        plt.show()
    return x_reduce

def fitness_std(generations: list[int], fitness_values: list[float], mean_value: list[float], std_values: list[float]) -> None:
    """
    Plots the fitness function along with its standard deviation for each generation.
    :param generations: List of generations (x-axis).
    :param fitness_values: List of fitness values for each generation.
    :param mean_value: List of mean fitness values for each generation.
    :param std_values: List of standard deviation values for each generation.
    :return: None
    """
    # Convert inputs to numpy arrays for easier manipulation
    generations = npArray(generations)
    fitness_values = npArray(fitness_values)
    mean_values = npArray(mean_value)
    std_values = npArray(std_values)
    # Calculate upper and lower bounds for the shaded region (fitness ± std deviation)
    upper_bound = mean_values + std_values
    lower_bound = mean_values - std_values
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_values, label="Best Fitness", color="green", linewidth=3)
    plt.plot(generations, mean_values, label="Mean Fitness", color="blue", linewidth=2)
    # Add shaded area for standard deviation
    plt.fill_between(generations, lower_bound, upper_bound, color='blue', alpha=0.3, label="Mean Fitness ± STD")
    # Labels and title
    plt.title("Convergence")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Value")
    # Show legend
    plt.legend(loc='lower right')
    # Display the plot
    plt.show()

def fitness_min(generations: list[int], fitness_values: list[float], mean_value: list[float], min_value: list[float]) -> None:
    """
    Plots the fitness function along with its standard deviation for each generation.
    :param generations: List of generations (x-axis).
    :param fitness_values: List of fitness values for each generation.
    :param mean_value: List of mean fitness values for each generation.
    :param std_values: List of standard deviation values for each generation.
    :return: None
    """
    # Convert inputs to numpy arrays for easier manipulation
    generations = npArray(generations)
    fitness_values = npArray(fitness_values)
    mean_values = npArray(mean_value)
    min_values = npArray(min_value)
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_values, label="Max Fitness", color="green", linewidth=3)
    plt.plot(generations, mean_values, label="Mean Fitness", color="blue", linewidth=2)
    plt.plot(generations, min_values, label="Min Fitness", color="red", linestyle="--", linewidth=2)
    # Labels and title
    plt.title("Convergence")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Value")
    # Show legend
    plt.legend(loc='lower right')
    plt.tight_layout()
    # Display the plot
    plt.show()

def exec_time(generations: list[int], times: list[float], elapsed_times: list[float]) -> None:
    """
    Plots the execution time for each generation.
    :param generations: List of generations (x-axis).
    :param times: List of execution time (Cumulative) for each generation.
    :param elapsed_times: List of elapsed time per generation.
    :return: None
    """
    # Convert inputs to numpy arrays for easier manipulation
    generations = npArray(generations)
    times_values = npArray(times)
    elapsed_times_values = npArray(elapsed_times)
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, times_values, label="Cumulative time", color="green", linewidth=3)
    plt.plot(generations, elapsed_times_values, label="Execution time", color="blue", linewidth=2, linestyle="--")
    # Labels and title
    plt.title("Execution Time")
    plt.xlabel("Generations")
    plt.ylabel("Time (s)")
    # Show legend
    plt.legend(loc='best')
    plt.tight_layout()
    # Display the plot
    plt.show()

def visualize_liquid(lsm, name: str = "Liquid visualization") -> None:
    """
    Visualizes the neurons in the LSM (only liquid).
    :param lsm: An instance of the LSM class, containing neuronsE and neuronsI attributes.
    :param name: Name of the plot
    :return: None
    """
    # Start the plot
    plt.figure(figsize=(10, 8))
    plt.title(name)
    # Get Global id's (GIDs) and Voltage thresholds for excitatory and inhibitory neurons
    exc_gids: list = list(lsm.neuronsE.get("global_id"))
    inh_gids: list = list(lsm.neuronsI.get("global_id"))
    # Get positions of excitatory and inhibitory neurons
    exc_pos: list = [GetPosition(gid) for gid in lsm.neuronsE]
    inh_pos: list = [GetPosition(gid) for gid in lsm.neuronsI]
    # Create dictionary with GIDs: position
    gids_pos: dict[int, list[float, float]] = {gid: gid_pos for gid, gid_pos in zip(exc_gids + inh_gids, exc_pos + inh_pos)}
    # Recover connections
    conn_In: SynapseCollection = GetConnections(synapse_model='Input')
    conn_E: SynapseCollection = GetConnections(synapse_model='e_syn')
    conn_I: SynapseCollection = GetConnections(synapse_model='i_syn')
    # Check if inhibitory connections are empty. Only check this since is more likely to be empty
    if not conn_I:
        check_connections: list = [*conn_E]
    else:
        check_connections: list = [*conn_E, *conn_I]
    # Plot Connections
    for connection in check_connections:
        # Get source and target position
        s_pos = gids_pos[connection.source]
        t_pos = gids_pos[connection.target]
        # Get type of connection
        type_c = "green" if connection.synapse_model == "e_syn" else "red"
        # Draw connections
        # plt.plot([s_pos[0], t_pos[0]], [s_pos[1], t_pos[1]], color=type_c, alpha=0.5)
        # Draw directional arrow for each connection
        plt.annotate('', xy=t_pos, xytext=s_pos, arrowprops=dict(arrowstyle="->", color=type_c, alpha=0.5))
    # Identify excitatory neurons that receive input connection
    in_exc: set = set()
    for in_conn in conn_In:
        in_exc.add(in_conn.target)  # Add target neuron to the set
    # Extract x and y coordinates for excitatory and inhibitory neurons
    exc_in_x, exc_in_y = zip(*[gids_pos[gid] for gid in exc_gids if gid in in_exc])
    # If there are any excitatory neurons without Input connection, extract their coordinates.

    inh_x, inh_y = zip(*[gids_pos[gid] for gid in inh_gids])
    # Plot neurons
    plt.scatter(exc_in_x, exc_in_y, color='green', marker="s", label='Excitatory (Input connected)', s=100)
    plt.scatter(inh_x, inh_y, color='red', label='Inhibitory', s=100)
    # If there are any excitatory neurons without Input connection, extract their coordinates and plot them.
    if len(in_exc) != len(exc_gids):
        exc_x, exc_y = zip(*[gids_pos[gid] for gid in exc_gids if gid not in in_exc])
        plt.scatter(exc_x, exc_y, color='green', label='Excitatory', s=100)
    # Add neuron GID labels for all neurons
    for gid, (x, y) in gids_pos.items():
        plt.text(x, y, f"{gid}", fontsize=8, ha='center', va='center', color="black")
    # Add labels and legend
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="best")
    plt.show()

def visualize_positions(encoding, name: str = "Position visualization") -> None:
    """
    Visualizes the position of neurons in the Liquid..
    :param encoding: An instance of the encoding class, containing the positions and polarities.
    :param name: Name of the plot
    :return: None
    """
    # Get x & y position of excitatory neurons
    x_exc = [encoding.positions[i][0] for i in encoding.indexesE]
    y_exc = [encoding.positions[i][1] for i in encoding.indexesE]
    # Get x & y position of inhibitory neurons
    x_inh = [encoding.positions[i][0] for i in encoding.indexesI]
    y_inh = [encoding.positions[i][1] for i in encoding.indexesI]
    # Plot
    plt.figure(num=name, figsize=(10, 8))
    plt.title(name)
    plt.scatter(x_exc, y_exc, color='green', label='Excitatory', s=100)
    plt.scatter(x_inh, y_inh, color='red', label='Inhibitory', s=100)
    # Add labels and legend
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="best")
    plt.grid(visible=True)
    plt.show(block=True)

def plot_neuron_density(encoding, name = "Neuron Positions") -> None:
    """
    Visualizes the neurons in the Liquid.
    :param encoding: An instance of the encoding class, containing the positions and polarities.
    :param name: Name of the plot
    :return: None
    """
    # Get x & y positions
    x_exc = [encoding.positions[i][0] for i in encoding.indexesE]
    y_exc = [encoding.positions[i][1] for i in encoding.indexesE]
    x_inh = [encoding.positions[i][0] for i in encoding.indexesI]
    y_inh = [encoding.positions[i][1] for i in encoding.indexesI]
    # Set up plot
    plt.figure(num=name, figsize=(10, 8))
    plt.title(f"{name}")
    # Scatter plot for neurons
    plt.scatter(x_exc, y_exc, color='green', label='Excitatory', s=100)
    plt.scatter(x_inh, y_inh, color='red', label='Inhibitory', s=100)
    # Apply dimension limits
    plot_dim = max(1, encoding.dim)  # Ensure dimension of at least 1 for smaller values
    plt.xlim(0, plot_dim)
    plt.ylim(0, plot_dim)
    # Add grid lines spaced at 1 unit apart
    plt.grid(visible=True, which='both', linestyle='--', color='grey', linewidth=0.5)
    plt.xticks(ticks=range(int(plot_dim) + 1))
    plt.yticks(ticks=range(int(plot_dim) + 1))
    # Add labels and legend
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def plot_input_classes(classes: list[list], name: str = "Input classes") -> None:
    """
    Plot input classes in the same figure.
    :param classes: Classes to plot.
    :param name: Name of the plot
    :return: None
    """
    # Get number of classes and channels
    num_classes = len(classes)
    num_channels = len(classes[0])
    # Define subplot grid dimensions
    cols = 2  # Number of columns in the subplot grid
    rows = (num_classes + cols - 1) // cols  # Calculate rows needed
    # Prepare the figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True, sharey=True)
    fig.suptitle(name, fontsize=16)
    # Flatten axes for easy iteration, in case rows * cols > n_neurons
    axes = axes.flatten()
    for i, c in enumerate(classes):
        # Iterate over information
        for channel, spike_times in enumerate(c):
            _channel = [channel for _ in range(len(spike_times))]
            axes[i].plot(spike_times, _channel, "|", c="black")
        axes[i].set_title(f"Class {i}")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Channels")
        axes[i].set_yticks(range(num_channels))
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.show()

def plot_PR_visualization(classes: list[list], instances: list[list],  name: str = "Input classes") -> None:
    """
    Visualization of an example for Pattern recognition task.
    :param classes: Classes to plot.
    :param instances: Instances to plot.
    :param name: Name of the plot
    :return: None
    """
    # Color instances
    colors = ["red", "green", "blue"]
    # Get number of classes and channels
    num_classes = len(classes)
    num_channels = len(classes[0])
    # Define subplot grid dimensions
    cols = 2  # Number of columns in the subplot grid
    rows = (num_classes + cols - 1) // cols  # Calculate rows needed
    # Prepare the figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True, sharey=True)
    fig.suptitle(name, fontsize=16)
    # Flatten axes for easy iteration, in case rows * cols > n_neurons
    axes = axes.flatten()
    for i, c in enumerate(classes):
        # Iterate over Patterns
        set_label = True
        for channel, spike_times in enumerate(c):
            _channel = [channel for _ in range(len(spike_times))]
            if set_label:
                axes[i].scatter(spike_times, _channel, marker="|", s=30, c="black", label="Original Patterns")
                set_label = False
            else:
                axes[i].scatter(spike_times, _channel, marker="|", s=30, c="black")
        # Show instances variations (3 variations)
        for n, j in zip([0, 1, 2], range(i, len(instances), 3)):
            set_label = True
            for channel, spike_times in enumerate(instances[j]):
                _channel = [channel for _ in range(len(spike_times))]
                if set_label:
                    axes[i].scatter(spike_times, _channel, marker=".", s=10, c=colors[n], label=f"Instance {j}")
                    set_label = False
                else:
                    axes[i].scatter(spike_times, _channel, marker=".", s=10, c=colors[n])
        axes[i].set_title(f"Class {i}")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Channels")
        axes[i].set_yticks(range(num_channels))
        axes[i].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.show()










