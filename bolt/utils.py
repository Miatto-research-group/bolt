from typing import Tuple, List, Generator, Dict
from itertools import product
import numpy as np
from functools import lru_cache
from collections import defaultdict
from numba.typed import Dict as NumbaDict
from numba.types import UniTuple, int64, complex128
# from numba.np.unsafe.ndarray import to_fixed_tuple
from numba.cpython.unsafe.tuple import tuple_setitem
from numba import jit

sqrt = np.sqrt(np.arange(100, dtype=np.complex64))

@lru_cache
def partition(photons:int, max_vals:Tuple[int]) -> List[Tuple[int]]:
    "a list for all the ways of putting n photons into modes that have at most (n1, n2, etc.) photons each"
    return [comb for comb in product(*(range(min(photons, i) + 1) for i in max_vals)) if sum(comb)==photons]


def depth_cost(scan_pattern:Tuple[int]) -> List[int]:
    """
    A list with the number of amplitudes we need to compute to reach each new level down the tree.
    It depends on the pattern that we need to scan, not the one we need to build.
    """
    photons = sum(scan_pattern)
    half = [len(tuple(partition(n, scan_pattern))) for n in range(photons//2+1)]
    return (half + list(reversed(half[:(photons+1)//2])))[1:-1]


def approx_tree_cost(build_patterns:List[Tuple[int]], scan_patterns:List[Tuple[int]]) -> int:
    "the approximated cost of building the whole tree for given lists of build and scan patterns"
    return sum(len(build_patterns)*sum(depth_cost(scan)) for scan in scan_patterns)


@jit(cache=True, nopython=True)
def remove(pattern:tuple) -> Tuple[int, Tuple[int]]:
    for p,n in enumerate(pattern):
        copy = pattern[:]
        if n > 0:
            yield p, tuple_setitem(copy, p, pattern[p] - 1) # pylint: disable=no-value-for-parameter


def build_order(kbuild:Tuple[int], num_modes) -> Tuple[Tuple[int], Tuple[int], int]:
    current_kbuild = [0 for _ in range(num_modes)]
    prev_kbuild = [0 for _ in range(num_modes)]
    for i,k in enumerate(kbuild):
        while current_kbuild[i] < k:
            current_kbuild[i] += 1
            yield tuple(prev_kbuild), tuple(current_kbuild), i
            prev_kbuild[i] += 1


@jit(cache=True, nopython=True)
def add_photon_to_output(kin:Tuple[int], koutplus1:int, i:int, V:np.array, U:NumbaDict, dU:NumbaDict, grad:bool): 
    """
    Implements the recurrence relation where we add a photon to the output

    Arguments:
        kin: input tuple
        koutplus1: k^out_i+1 in formula 
        i: mode where we add the photon
        V: current covariance matrix
        U: current U_k^out dict
        dU current dU_k^out dict
    """
    _U = 0.0 + 0.0j
    for p,kin_p in remove(kin):
        _U += sqrt[kin[p]]*V[i,p]*U[kin_p]
    _U /= sqrt[koutplus1]
    _dU = None
    if grad:
        _dU = np.zeros_like(V)
        for p,kin_p in remove(kin):
            _dU += sqrt[kin[p]] * V[i,p] * dU[kin_p]
            _dU[i,p] += sqrt[kin[p]]*U[kin_p]
        _dU /= sqrt[koutplus1]
    return _U, _dU

@jit(cache=True, nopython=True)
def add_photon_to_input(kout:Tuple[int], kinplus1:Tuple[int], i:int, V:np.array, U:NumbaDict, dU:NumbaDict, grad:bool): 
    """
    Implements the recurrence relation where we add a photon to the input

    Arguments:
        kout: output tuple
        kinplus1: k^in_i+1 in formula 
        i: mode where we add the photon
        V: current covariance matrix
        U: current U_k^in dict
        dU current dU_k^in dict
    """
    _U = 0.0 + 0.0j
    for p,kout_p in remove(kout):
        _U += sqrt[kout[p]]*V[p,i]*U[kout_p]
    _U /= sqrt[kinplus1]
    _dU = None
    if grad:
        _dU = np.zeros_like(V)
        for p,kout_p in remove(kout):
            _dU += sqrt[kout[p]] * V[p,i] * dU[kout_p]
            _dU[p,i] += sqrt[kout[p]]*U[kout_p]
        _dU /= sqrt[kinplus1]
    return _U, _dU

def all_outputs(state_in, V, grad=True):
    num_modes = state_in.num_modes

    U = defaultdict(lambda: NumbaDict.empty(key_type=UniTuple(int64, num_modes), value_type=complex128))
    U[(0,)*num_modes][(0,)*num_modes] = 1.0 + 0.0j
    dU = defaultdict(lambda: NumbaDict.empty(key_type=UniTuple(int64, num_modes), value_type=complex128[:,:]))
    dU[(0,)*num_modes][(0,)*num_modes] = np.zeros_like(V, dtype=np.complex128)

    for kin in state_in:
        for prev_kbuild, current_kbuild, mode in build_order(kin, num_modes):
            photons = sum(current_kbuild)
            for _kscan in partition(photons, (photons,)*num_modes):
                if _kscan not in U[current_kbuild]:
                    u,du = add_photon_to_output(_kscan, current_kbuild[mode], mode, V, U[prev_kbuild], dU[prev_kbuild], grad)
                    U[current_kbuild][_kscan] = u
                    if grad:
                        dU[current_kbuild][_kscan] = du
    
    out = defaultdict(complex)
    dout = defaultdict(complex)

    for kin,amp in state_in.items():
        for kout in U[kin]:
            out[kout] += amp*U[kin][kout]
            dout[kout] += amp*dU[kin][kout]

    return out, dout
    
