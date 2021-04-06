import numpy as np
from .tree import Tree
from .states import State, IOSpec
from .utils import partition

class Interferometer:
    """
    Interferometer class. Give you a specific covariance matrix for an alternate interferometer and an input State, you can find out all possible output states with its amplitudes.

        Usage:
        >>> intfrmt = Interferometer(cov_matrix = S)
        >>> intfrmt.getalloutputs(inputstate = state_in)
    """
    def __init__(self, cov_matrix):
        self.cov_matrix = cov_matrix
        
    def set_covmatrix(self, cov):
        self.cov_matrix = cov
        
    def get_covmatrix(self):
        return self.cov_matrix
        
    def getalloutputs(self, inputstate:State):
        """
        Given a state input, you can get all possible outputs and its amplitudes.
        
        Arguments:
            inputstate (State): State object
        Returns:
            Dict[state]: amplitude
        """
        num_photons = sum(list(inputstate.keys())[0])
        num_modes = len(list(inputstate.keys())[0])
        if num_modes != len(self.cov_matrix):
            raise ValueError(f"""The input state does not have the same number of modes as the given covariance matrix.
        Solution: Please modify.""")
        list_possible_output = partition(num_photons,(num_modes,)*num_modes)
        alloutputs = {}
        for i in range(len(list_possible_output)):
            output = list_possible_output[i]
            io = IOSpec(inputstate, State({output:1}))
            t1 = Tree(io=io, covariance_matrix=self.cov_matrix, grad=True)
            amp,dU = t1.amplitude()
            alloutputs[output] = amp
        return alloutputs
