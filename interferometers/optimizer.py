import time
import numpy as np
from numpy_ml.neural_nets.optimizers import Adam
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
    def __init__(self, lr:float, epsilon:float = 1e-5, max_steps:int = 10000):
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.opt = Adam(lr=lr)
        self.losses = []
        self.elapsed = 0.0
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
        lambdas = np.random.normal(size=req.modes**2, scale=1)
        V = expm(L(lambdas))
        self.losses = [1e6]
        try:
            progress = trange(self.max_steps)
            for step in progress:
                grad_update = 0
                loss = 0
                probs = []
                for io,prob in req.specs.items():
                    tree = Tree(io=io, covariance_matrix=V, grad=True)
                    a, da_dV = tree.amplitude()
                    p = abs(a)**2
                    probs.append(p)
                    dL_da = np.conj(a)*(p - prob) - prob**2/a
                    da_dlambdas = np.sum(da_dV * dV_dlambdas(lambdas), axis=(1,2))
                    grad_update += 2*np.real(dL_da * da_dlambdas)
                    loss += self.mse(p, prob) - prob*np.log(p/prob)
                self.losses.append(loss)
                lambdas = self.opt.update(lambdas, grad_update, None, None)
                V = expm(L(lambdas))
                P = ', '.join([f'{p:.4f}' for p in probs])
                progress.set_description(f"{step}: probs = [{P}], loss = {loss:.4f}")
                if abs(self.losses[-2] - self.losses[-1]) < self.epsilon:
                    break
        except KeyboardInterrupt:
            print('gracefully stopping optimization...')
        self.losses.pop(0)
        toc = time.time()
        self.elapsed = toc-tic
        return V



