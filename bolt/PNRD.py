import time
import numpy as np
from numpy_ml.neural_nets.optimizers import Adam
from tqdm import trange

# from scipy.linalg import expm
from typing import Tuple, List
from .expm import expm
from .tree import Tree
from .liealgebra import L, dV_dlambdas
from .states import State,IOSpec,Requirements
from .utils import partition,pos,all_outputs

class PNRD_State_Discriminator:
    """
    Interferometer optimizer class. Create an instance by specifying:
        lr (float): learning rate
        epsilon (float): the optimization keeps going until the drop in loss is larger than epsilon
        max_steps: (int): stop at max_steps if the epsilon criterion is not met yet

        Usage:
        >>> opt = Optimizer(lr=0.01, epsilon=1e-4)
        >>> cov_matrix = opt(requirements)
    """
    def __init__(self, lr:float, epsilon:float = 1e-6, max_steps:int = 1000, cov_matrix_init=None, natural:bool = False):
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.losses = []
        self.probs = []
        self.elapsed = 0.0
        self.cov_matrix_init = cov_matrix_init
        self.natural = natural
        if self.natural:
            self.lr = lr
        else:
            self.lr = lr
            self.opt = Adam(lr=lr)

    @staticmethod
    def mse(x, y):
        return 0.5*(x-y)**2
    @staticmethod
    def no_photons(x:State):
        z=0
        for y in x:
            z=y
            break
        z=list(z)
        x=0
        for i in z:
            x+=i
        return x
    def __call__(self, st:[State]):
        """
        The optimizer instance supports being called as a function on a Requirements object.
        It records the list of losses (self.losses) and the elapsed time (self.elapsed).
        
        Arguments:
            req (Requirements): requirements object
        Returns:
            covariance matrix
        """
        lengthst=int(len(st))
        num_modes=st[0].num_modes
        num_photons=self.no_photons(x=st[0])
        size=st[0].num_modes**2
        states=partition(num_photons,(num_photons,)*num_modes)
        tic = time.time()
        if self.cov_matrix_init:
            V = self.cov_matrix_init
        else:
            lambdas = np.random.normal(size=size, scale=0.01)
            V = expm(L(lambdas))
        if self.natural:
            A = np.block([[np.real(V), -np.imag(V)],[np.imag(V), np.real(V)]])
        self.losses = [1e6]
        try:
            progress = trange(self.max_steps)
            for step in progress:
                grad_update = 0
                loss = 0
                op=[]
                op_g=[]
                for i in range(lengthst):
                    u,v= all_outputs(st[i], V, grad=True)
                    # op is an array of dictionaries, each element being the corresponding output state for each input state 
                    op.append(u)
                    op_g.append(v)
                for i in range(lengthst):
                    for j in range(lengthst):
                        if(j>i):
                            for r in states:
                                a1 = op[i].get(r)
                                da1_dV = op_g[i].get(r)
                                a2 = op[j].get(r)
                                da2_dV = op_g[j].get(r)
                                p1 = abs(a1)**2
                                p2 = abs(a2)**2
                                loss += p1*p2
                                dL_da1=np.conj(a1)*p2
                                dL_da2=np.conj(a2)*p1
                                if self.natural:
                                    da1_dA = np.block([[da1_dV, -1j*da1_dV],[1j*da1_dV, da1_dV]])
                                    da2_dA = np.block([[da2_dV, -1j*da2_dV],[1j*da2_dV, da2_dV]])
                                    grad_update += 2*np.real((dL_da1 * da1_dA)+(dL_da2 * da2_dA))
                                else:
                                    da1_dlambdas = np.sum(da1_dV * dV_dlambdas(lambdas), axis=(1,2))
                                    da2_dlambdas = np.sum(da2_dV * dV_dlambdas(lambdas), axis=(1,2))
                                    grad_update += 2*np.real((dL_da1 * da1_dlambdas)+(dL_da2 * da2_dlambdas))
                #print(grad_update)
                self.losses.append(loss)
                if self.natural:
                    Q = grad_update
                    D = 0.5*(Q - A @ Q.T @ A) # natural gradient
                    A = A @ expm(self.lr * D.T @ A)
                    V = A[:len(V), :len(V)] + 1j*A[len(V):, :len(V)]
                else:
                    lambdas = self.opt.update(lambdas, grad_update, None, None)
                    V = expm(L(lambdas))
                progress.set_description(f"{step}:loss = {loss:.4f}")
                if abs(self.losses[-2] - self.losses[-1]) < self.epsilon:
                    break
        except KeyboardInterrupt:
            print('gracefully stopping optimization...')
        self.losses.pop(0)
        toc = time.time()
        self.elapsed = toc-tic
        return V