from __future__ import annotations
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations, product
from typing import Tuple, Dict, List
from .expm import expm


from joblib import Parallel, delayed
import dill as pickle

from numba.typed import Dict
from numba.types import UniTuple, int64, complex128

from .states import IOSpec
from .utils import partition, remove, build_order, add_photon_to_output, add_photon_to_input
from .liealgebra import L


class Tree:
    """
    The algorithm that computes input-output interferometer amplitudes builds a tree.
    We can reuse the tree in case of multiple input-output relations. This class supports all that.
    """
    def __init__(self, io:IOSpec, covariance_matrix:np.array = None, grad:bool = True):
        if covariance_matrix is None:
            modes = len(list(io.input.keys())[0])
            lambdas = np.random.normal(size=modes**2, scale=1)
            covariance_matrix = expm(L(lambdas))
        self.V = covariance_matrix
        self.num_modes = self.V.shape[0]
        self.io = io
        self.grad = grad
        self.reset(self.num_modes)
        
    def reset(self, size):
        self.U = defaultdict(lambda: Dict.empty(key_type=UniTuple(int64, size), value_type=complex128))
        self.U[(0,)*size][(0,)*size] = 1.0 + 0.0j

        self.dU = defaultdict(lambda: Dict.empty(key_type=UniTuple(int64, size), value_type=complex128[:,:]))
        self.dU[(0,)*size][(0,)*size] = np.zeros_like(self.V, dtype=np.complex128)

    def build_tree(self, kbuild:Tuple[int], kscan:Tuple[int], building_input:bool):
        "builds the tree by adding photons either to the input or to the output modes"
        for prev_kbuild, current_kbuild, mode in build_order(kbuild, self.num_modes):
            photons = sum(current_kbuild)
            for _kscan in partition(photons, kscan):
                if _kscan not in self.U[current_kbuild]: # not in dU[current_kbuild] either
                    if building_input:
                        U,dU = add_photon_to_input(_kscan, current_kbuild[mode], mode, self.V, self.U[prev_kbuild], self.dU[prev_kbuild], self.grad)
                    else:
                        U,dU = add_photon_to_output(_kscan, current_kbuild[mode], mode, self.V, self.U[prev_kbuild], self.dU[prev_kbuild], self.grad)
                    self.U[current_kbuild][_kscan] = U
                    if self.grad: self.dU[current_kbuild][_kscan] = dU

    def step(self, tup):
        kbuild, kscan = tup
        self.build_tree(kbuild, kscan, self.io.building_input)

    def amplitude(self):
        U = 0
        dU = np.zeros_like(self.V)
        for kbuild,kscan,amp in self.io.paths: # TODO this i can parallelize debbrutto
            self.build_tree(kbuild, kscan, self.io.building_input)
            U += amp * self.U[kbuild][kscan]
            if self.grad:
                dU += amp * self.dU[kbuild][kscan]
        if self.grad:
            return U, dU
        else:
            return U


    # def amplitude(self):
    #     U = 0
    #     dU = np.zeros_like(self.V)
    #     with nogil:
    #         Parallel(n_jobs=6, require='sharedmem')(delayed(self.step)((kbuild, kscan)) for kbuild,kscan,amp in self.io.paths)
    #     for kbuild,kscan,amp in self.io.paths:
    #         U += amp * self.U[kbuild][kscan]
    #         if self.grad:
    #             dU += amp * self.dU[kbuild][kscan]
    #     if self.grad:
    #         return U, dU
    #     else:
    #         return U

    