from collections import defaultdict
from itertools import combinations
import numpy as np
from typing import Tuple, Dict
from states import State, IOSpec, Requirements


def drop_single(tup):
    """
    A generator for the pairs (el, tup.drop(el)) for each element in tup
    """
    for i in range(len(tup)):
        yield tup[i], tup[:i] + tup[i+1:]

def drop_multi(tup):
    """
    A generator for the triples (multiplicity(el), el, tup.drop(el)) for each unique element in tup
    """
    multiplicity = defaultdict(int)
    index = dict()
    for i,t in enumerate(tup):
        if t not in multiplicity:
            index[t] = i
        multiplicity[t] += 1
    for t in multiplicity:
        i = index[t]
        yield multiplicity[t], t, tup[:i] + tup[i+1:]

def amplitude_and_grad(input, output, V):
    """
    Returns the input-output amplitude of the interferometer given
    by the covariance matrix V and its gradient with respect to V
    It accounts for any photon number in any mode.
    Arguments:
        input (tuple): input pattern in s-notation
        output (tuple): output pattern in s-notation
    """
    output_count = defaultdict(int)
    U = {tuple() : 1.0+0.0j}
    dU = {tuple() : np.zeros_like(V, dtype=V.dtype)}
    sqrt = np.sqrt(np.arange(sum(output)))
    for step,i in enumerate(output):
        output_count[i] += 1
        newU = defaultdict(np.complex64)
        newdU = defaultdict(np.complex64)
        for key in set(combinations(input, step+1)):
            for m,p,tup in drop_multi(key):
                newU[key] += sqrt[m]*V[i,p]*U[tup]
                newdU[key] += sqrt[m]*V[i,p]*dU[tup]
                newdU[key][i,p] += sqrt[m]*U[tup]
            newU[key] /= sqrt[output_count[i]]
            newdU[key] /= sqrt[output_count[i]]
        U = newU
        dU = newdU
    return U[input], dU[input]


def amplitude_multiinput_and_grad(input_multi:Dict[Tuple[int],complex], output:Tuple[int], V):
    '''
    Returns the input-output amplitudes for multiple input patterns of the interferometer
    given by the covariance matrix V and their gradient with respect to V.
    Generallly faster than calling amplitude_and_grad() multiple times.
    Arguments:
        input_multi (dict(tuple, complex)): dictionary of patterns in s-notation 
            with corresponding amplitudes to specify input state
        output (tuple): pattern of the output in s-notation
    '''
    U = defaultdict(lambda: defaultdict(np.complex64))
    dU = defaultdict(lambda: defaultdict(lambda: np.zeros_like(V, dtype=V.dtype)))
    U[tuple()][tuple()] = 1.0 + 0.0j
    sqrt = np.sqrt(np.arange(100))
    for input_ in input_multi:
        output_count = defaultdict(int)
        index = []
        for step,i in enumerate(input_):
            index.append(i)
            output_count[i] += 1
            tup_index = tuple(index)
            if tup_index not in U: # not in dU either then
                for key in set(combinations(output, step+1)):
                    for m,p,tup in drop_multi(key):
                        U[tup_index][key] += sqrt[m]*V[p,i]*U[tup_index[:-1]][tup]
                        dU[tup_index][key] += sqrt[m]*V[p,i]*dU[tup_index[:-1]][tup]
                        dU[tup_index][key][p,i] +=  sqrt[m]*U[tup_index[:-1]][tup]
                    U[tup_index][key] /= sqrt[output_count[i]]
                    dU[tup_index][key] /= sqrt[output_count[i]]
    return sum(amp_*U[input_][output] for input_, amp_ in input_multi.items()), sum(amp_*dU[input_][output] for input_,amp_ in input_multi.items())


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

def dV_dlambdas(lambdas):
    'returns the gradient of the interferometer matrix with respect to the Lie algebra basis'
    n = int(np.sqrt(len(lambdas)))
    Vs = []
    d,W = np.linalg.eigh(1j*L(lambdas))
    d = -1j*d
    E = np.exp(d)
    ED = (E[:,None] - E[None,:])/(d[:,None] - d[None,:] + 1e-9) + np.diag(E)

    for a in range(n):
        WTW = 1j*np.outer(np.conj(W[a]), W[a])
        Vs.append(np.linalg.multi_dot([W, WTW*ED, np.conj(W.T)]))
    for s in range(1,n):
        for r in range(s):
            WTW = 1j*(np.outer(np.conj(W[r]), W[s]) + np.outer(np.conj(W[s]), W[r]))
            Vs.append(np.linalg.multi_dot([W, WTW*ED, np.conj(W.T)]))
    for s in range(1,n):
        for r in range(s):
            WTW = (np.outer(np.conj(W[r]), W[s]) - np.outer(np.conj(W[s]), W[r]))
            Vs.append(np.linalg.multi_dot([W, WTW*ED, np.conj(W.T)]))

    return np.array(Vs)