from collections import defaultdict
from itertools import combinations
import numpy as np
from typing import Tuple, Dict
from numba import njit

from .states import State, IOSpec, Requirements

@jit(cache=True, nopython=True)
def L(lambdas):
    'returns the Lie algebra element in the lambda basis'
    n = int(np.sqrt(len(lambdas))) # there are n^2 lambdas
    L = 1j*np.diag(lambdas[:n])
    c = n 
    for s in range(1,n):
        for r in range(s):
            L[s,r] += 1j*lambdas[c] - lambdas[c + n*(n-1)//2]
            L[r,s] += 1j*lambdas[c] + lambdas[c + n*(n-1)//2]
            c += 1
    return L

@jit(cache=True, nopython=True)
def dV_dlambdas(lambdas):
    'returns the gradient of the interferometer matrix with respect to the Lie algebra basis'
    n = int(np.sqrt(len(lambdas)))
    Vs = []
    d,W = np.linalg.eigh(1j*L(lambdas))
    d = -1j*d
    E = np.exp(d)
    ED = (np.expand_dims(E,1) - np.expand_dims(E,0))/(np.expand_dims(d,1) - np.expand_dims(d,0) + 1e-9)+ np.diag(E)

    for a in range(n):
        WTW = 1j*np.outer(np.conj(W[a]), W[a])
        Vs.append(W@(WTW*ED)@np.conj(W.T))
    for s in range(1,n):
        for r in range(s):
            WTW = 1j*(np.outer(np.conj(W[r]), W[s]) + np.outer(np.conj(W[s]), W[r]))
            Vs.append(W@(WTW*ED)@np.conj(W.T))
    for s in range(1,n):
        for r in range(s):
            WTW = (np.outer(np.conj(W[r]), W[s]) - np.outer(np.conj(W[s]), W[r]))
            Vs.append(W@(WTW*ED)@np.conj(W.T))

    stacked = np.zeros((n**2,n,n), dtype=np.complex128)
    for k,V in enumerate(Vs):
        stacked[k] = V

    return stacked