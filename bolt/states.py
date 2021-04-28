from typing import Dict, Tuple
from collections.abc import MutableMapping
import numpy as np
import random
from .utils import approx_tree_cost

class State(MutableMapping):
    """
    A dictionary that represents a pure quantum state.
    It stores ket:amplitude pairs and supports a `normalize()` method.
    
    Usage:
    >>> state = State(mydict) # mydict is a dictionary with ket:amplitude pairs
    or
    >>> state = State()
    >>> state[ket] = amplitude
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)
    
    def __len__(self):
        return len(self.store)

    def normalize(self):
        norm = sum(abs(s)**2 for s in self.store.values())
        for k in self.store:
            self.store[k] /= np.sqrt(norm)
    
    @property
    def num_modes(self):
        return len(random.sample(self.store.keys(),1)[0])

    def __repr__(self):
        strings = []
        for ket,amp in self.store.items():
            if np.real(amp) < 0:
                sign = '- '
                amp = -amp
            else:
                sign = '+ '
            if np.isclose(amp, 1.0):
                strings.append(sign + '|' + ','.join(map(str,ket)) + '> ')
            else:
                strings.append(sign + f'{amp:.4f}' + '|' + ','.join(map(str,ket)) + '> ')
        if '+' in strings[0][0]:
            strings[0] = strings[0][2:]
        return ''.join(strings)


class IOSpec:
    """
    Input-output specification.
    Supports generating all Input/Output pairs and their amplitude
    ---
    Arguments:
      input_state (State): input state
      output_state (State): output state
      build (str): "input", "output" or None (default None). Force Tree to build input or output or let IOSpec optimize it automatically.
                    For large inputs it may be slow at computing the optimal choice, and it could be intuitive which one is best.
    """
    def __init__(self, input_state:State, output_state:State, build:str = None):
        if len(set([sum(ket) for ket in input_state.keys()])) > 1:
            raise ValueError(f"""The input state spans more than one multiplet (photons per ket are {[sum(ket) for ket in input_state.keys()]}).
        Solution: separate the required input-output relations per multiplet and submit them as different requirements.""")
        if len(set([sum(ket) for ket in output_state.keys()])) > 1:
            raise ValueError(f"""The output state spans more than one multiplet (photons per ket are {[sum(ket) for ket in output_state.keys()]}).
        Solution: separate the required input-output relations per multiplet and submit them as different requirements.""")
        if sum(list(input_state.keys())[0]) != sum(list(output_state.keys())[0]):
            raise ValueError(f"""This spec does not conserve photon number
        (input photons = {sum(list(input_state.keys())[0])}, output photons = {sum(list(output_state.keys())[0])}).""")
        if len(set([len(ket) for ket in input_state.keys()])) > 1:
            raise ValueError(f"not all input states span the same number of modes (modes = {[len(ket) for ket in input_state.keys()]})")
        if len(set([len(ket) for ket in output_state.keys()])) > 1:
            raise ValueError(f"not all output states span the same number of modes (modes = {[len(ket) for ket in output_state.keys()]})")
        self.input = input_state
        self.output = output_state
        if build is None:
            cost_of_building_input  = approx_tree_cost(build_patterns=list(self.input.keys()), scan_patterns=list(self.output.keys()))
            cost_of_building_output = approx_tree_cost(build_patterns=list(self.output.keys()), scan_patterns=list(self.input.keys()))
        build = 'input' if cost_of_building_input < cost_of_building_output else 'output'
        if 'input' == build.lower():
            self.building_output = False
            self.building_input = True
            self.paths = [(ket_in, ket_out, self.input[ket_in]*np.conj(self.output[ket_out])) for ket_out in output_state for ket_in in input_state]
        elif 'output' in build.lower():
            self.building_output = True
            self.building_input = False
            self.paths = [(ket_out, ket_in, self.input[ket_in]*np.conj(self.output[ket_out])) for ket_out in output_state for ket_in in input_state]
        else:
            raise ValueError('argument build can be "input", "output" or None')
    @property
    def photons(self):
        return sum(list(self.input.keys())[0])

    @property
    def modes(self):
        return len(list(self.input.keys())[0])

    def __repr__(self):
        return repr(self.input) + '  --->  ' + repr(self.output)



class Requirements:
    """
    Collection of all the IO specifications for a given input state with corresponding weights.
    ---
    Arguments:
      specs (dict): dictionary with IOSpec:weight pairs. Weights can be probabilities or just some floats.
                    If weights are larger than 1.0, they make the KL divergence weigh more
                    than the input-output spec in the loss function.
    """

    def __init__(self, specs:Dict[IOSpec,float]):
        if len(set([io.modes for io in specs.keys()])) > 1:
            raise ValueError(f'not all input-output relations span the same number of modes (modes = {[io.modes for io in specs.keys()]})')
        self.specs = specs 

    @property
    def modes(self):
        return list(self.specs.keys())[0].modes


    def __repr__(self):
        strings = []
        for io,prob in self.specs.items():
            strings.append(repr(io) + f'   weight = {prob:.4f}\n')
        return ''.join(strings)

    
