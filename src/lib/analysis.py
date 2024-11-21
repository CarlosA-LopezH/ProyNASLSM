# General imports
import pickle
import matplotlib.pyplot as plt
from numpy import mean as npMean, std as npStd
from scipy.stats import kruskal as spKruskal, mannwhitneyu as spMannwhitneyu, shapiro as spShapiro
from scikit_posthocs import posthoc_dunn
# Source imports
from .visualization import fitness_std

def plot_convergence_runs(id_method: str, num_runs: int, root: str = "../Experiments/Debugging/checkpoint/") -> None:
    """
    Plots the convergence of fitness function runs for multiple experiments.
    Comments by: ChatGPT
    :param id_method: A string representing the identifier of the method used (to distinguish between different methods).
    :param num_runs: The number of experiment runs to plot.
    :param root: The root directory path where the checkpoint files for each run are stored.
    :return: None
    """
    # Loop through each run from 1 to num_runs
    for i in range(1, num_runs + 1):
        # Open the checkpoint file for the current run, the file naming convention is root + id_method + "_<run_number>.chck"
        with open(f"{root}{id_method}_{i}.chck", "rb") as file:
            # Load the data from the checkpoint file using pickle
            data = pickle.load(file)
        # Extract the "Logbook" (which contains information about generations, fitness, etc.) from the loaded data
        log = data["Logbook"]
        # Select the generation numbers from the logbook
        gen = log.select("gen")
        # Temporary fix: if the generation numbers start from -1, adjust them to start from 0 by adding 1 to each generation
        if gen[0] == -1:
            gen = [g + 1 for g in gen]
        # Extract the "max" (best fitness), "avg" (average fitness), and "std" (standard deviation of fitness) for each generation
        best_fit = log.select("max")
        avg_fit = log.select("avg")
        std_fit = log.select("std")
        # Call a function named 'fitness_std' to plot the fitness values and their standard deviation
        fitness_std(gen, best_fit, avg_fit, std_fit)


def load_data(root: str, method_id: str, num_runs: int) -> list:
    """
    Loads the performance data for a given method across multiple runs.
    Src: ChatGPT
    """
    method_data = []
    for i in range(1, num_runs + 1):
        with open(f"{root}{method_id}_{i}.chck", "rb") as file:
            data = pickle.load(file)
        log = data["Logbook"]
        best_fit = log.select("max")  # Change this as per the metric you want to analyze
        method_data.append(best_fit[-1])  # Taking the last generation's best fitness
    return method_data


def check_normality(method_ids: list, method_data: dict) -> None:
    """
    Perform Shapiro-Wilk test to check normality of fitness data for each method.
    Src: ChatGPT
    """
    for method_id in method_ids:
        stat, p_value = spShapiro(method_data[method_id])
        print(f"Shapiro-Wilk test for {method_id}: p-value = {p_value}")
        if p_value < 0.05:
            print(f"   {method_id} does not follow a normal distribution (reject H0).")
        else:
            print(f"   {method_id} follows a normal distribution (fail to reject H0).")


def compare_methods(method_ids: list, num_runs: int, root: str = "../Experiments/Debugging/checkpoint/") -> dict:
    """
    Automates the process of loading data, checking normality, performing statistical analysis, and plotting.
    Src: ChatGPT
    """
    method_data = {}
    # Load the data for each method
    for method_id in method_ids:
        method_data[method_id] = load_data(root, method_id, num_runs)
    # Perform normality checks
    print("Normality Check:")
    check_normality(method_ids, method_data)
    # Decide statistical test based on number of methods
    if len(method_ids) == 2:
        stat, p = spMannwhitneyu(method_data[method_ids[0]], method_data[method_ids[1]])
        print(f"Mann-Whitney U Test p-value for {method_ids[0]} vs {method_ids[1]}: {p}")
        if p < 0.05:
            print(f"Significant difference found between {method_ids[0]} and {method_ids[1]}")
        else:
            print(f"No significant difference between {method_ids[0]} and {method_ids[1]}")
    elif len(method_ids) > 2:
        data_values = [method_data[method_id] for method_id in method_ids]
        stat, p = spKruskal(*data_values)
        print(f"Kruskal-Wallis Test p-value for {method_ids}: {p}")
        if p < 0.05:
            posthoc_result = posthoc_dunn(data_values, p_adjust='bonferroni')
            print(f"Dunn's Test Results (Post-hoc):\n{posthoc_result}")
            # Identify pairs with significant differences
            for i in range(len(method_ids)):
                for j in range(i + 1, len(method_ids)):
                    if posthoc_result.iloc[i, j] < 0.05:
                        print(f"   Significant difference between {method_ids[i]} and {method_ids[j]} (p = {posthoc_result.iloc[i, j]})")
        else:
            print(f"No significant difference found among {method_ids}")
        # Plot the data
        plot_data(method_ids, method_data)
    return method_data


def plot_data(method_ids: list, method_data: dict) -> None:
    """
    Plots the fitness values for each method along with error bars for standard deviation.
    Src: ChatGPT
    """
    # Obtain mean and std for plotting
    means = [npMean(method_data[method_id]) for method_id in method_ids]
    stds = [npStd(method_data[method_id]) for method_id in method_ids]
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(method_ids, means, yerr=stds, capsize=5, width=0.1)
    # Perform plots
    plt.title("Fitness Comparison Among Methods")
    plt.xlabel("Methods")
    plt.ylabel("Fitness (Mean ± Std Dev)")
    plt.show()

if __name__ == "__main__":
    with open("../Experiments/Debugging/checkpoint/OLD/5C-FD_GA_BLX_KNN-FS_LDS_1.chck", "rb") as file:
        data = pickle.load(file)