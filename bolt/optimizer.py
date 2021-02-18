import time
import numpy as np
from numpy_ml.neural_nets.optimizers import Adam, SGD
from tqdm import trange

# from scipy.linalg import expm
from .expm import expm
from .tree import Tree
from .liealgebra import L, dV_dlambdas
from .states import Requirements

class Optimizer:
    """
    Interferometer optimizer class. Create an instance by specifying:
        lr (float): learning rate
        epsilon (float): the optimization keeps going until the drop in loss is larger than epsilon
        max_steps: (int): stop at max_steps if the epsilon criterion is not met yet

        Usage:
        >>> opt = Optimizer(lr=0.01, epsilon=1e-4)
        >>> cov_matrix = opt(requirements)
    """
    def __init__(self, lr:float, epsilon:float = 1e-5, max_steps:int = 1000, cov_matrix_init=None, natural:bool = False):
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.losses = []
        self.probs = []
        self.elapsed = 0.0
        self.cov_matrix_init = cov_matrix_init
        self.natural = natural
        if self.natural:
            self.opt = SGD(lr=lr)
        else:
            self.opt = Adam(lr=lr)

    @staticmethod
    def mse(x, y):
        return 0.5*(x-y)**2

    
    def __call__(self, req:Requirements):
        """
        The optimizer instance supports being called as a function on a Requirements object.
        It records the list of losses (self.losses) and the elapsed time (self.elapsed).
        
        Arguments:
            req (Requirements): requirements object
        Returns:
            covariance matrix
        """
        tic = time.time()
        if self.cov_matrix_init:
            V = self.cov_matrix_init
        else:
            lambdas = np.random.normal(size=req.modes**2, scale=0.01)
            V = expm(L(lambdas))
        if self.natural:
            A = np.block([[np.real(V), -np.imag(V)],[np.imag(V), np.real(V)]])

        self.losses = [1e6]
        try:
            progress = trange(self.max_steps)
            for step in progress:
                grad_update = 0
                loss = 0
                self.probs = []
                for io,prob in req.specs.items():
                    tree = Tree(io=io, covariance_matrix=V, grad=True)
                    a, da_dV = tree.amplitude()
                    p = abs(a)**2
                    self.probs.append(p)
                    dL_da = np.conj(a)*(p - prob) - prob**2/a

                    if self.natural:
                        da_dA = np.block([[da_dV, -1j*da_dV],[1j*da_dV, da_dV]])
                        grad_update += 2*np.real(dL_da * da_dA)
                    else:
                        da_dlambdas = np.sum(da_dV * dV_dlambdas(lambdas), axis=(1,2))
                        grad_update += 2*np.real(dL_da * da_dlambdas)
                    
                    loss += self.mse(p, prob) - prob*np.log(p/prob)
                self.losses.append(loss)
                if self.natural:
                    Q = grad_update
                    D = 0.5*(Q - A @ Q.T @ A) # natural gradient
                    t = self.opt.hyperparameters['lr']
                    A = A @ expm(t * D.T @ A) @ expm(t * (D.T @ A - A.T @ D))
                    V = A[:len(V), :len(V)] + 1j*A[len(V):, :len(V)]
                else:
                    lambdas = self.opt.update(lambdas, grad_update, None, None)
                    V = expm(L(lambdas))
                P = ', '.join([f'{p:.4f}' for p in self.probs])
                progress.set_description(f"{step}: probs = [{P}], loss = {loss:.4f}")
                if abs(self.losses[-2] - self.losses[-1]) < self.epsilon:
                    break
        except KeyboardInterrupt:
            print('gracefully stopping optimization...')
        self.losses.pop(0)
        toc = time.time()
        self.elapsed = toc-tic
        return V



