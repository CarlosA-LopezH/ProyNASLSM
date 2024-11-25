# General imports
from numpy.random import random as npRand, randint as npRandInt
from random import uniform
from math import ceil

class Encoding:
    """
    Encoding method. It contains the configurations and position of neurons.
    """
    # The following constants are necessary for inner procedures. They are not intended to be modified outside of it.
    # The original values are at the middle of the options. These are not based on any reference.
    _T_REF: list[float] = [0.5, 1., 2., 3., 4.]  # Duration of refractory period (ms).
    _TAU_M: list[float] = [6., 8., 10., 12., 14.]  # Membrane time constant (ms).
    _V_TH: list[float] = [-69., -55., -40., -25., -10.]  # Spike threshold (mV).
    _V_RESET: float = -70.  # Reset potential of the membrane (mV).
    _PROB_E: float = 0.8  # Probability for setting the excitatory polarity. Based on "Neuron Dynamics"
    def __init__(self, n_neurons: int, channels: int, dim: float = 6.) -> None:
        self.dim: float = ceil(dim) # The size (dimension) of the space for neuron positions. With bigger dimension, the distances grow
        # General parameters
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
        # Initialize random seed.
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
                         'polarity': 'E' if npRand() < self._PROB_E else 'I'}
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
