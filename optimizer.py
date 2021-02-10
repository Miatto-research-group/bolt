import numpy as np
from tree import Tree
from numpy_ml.neural_nets.optimizers import Adam
from states import Requirements
from tqdm import trange
from scipy.linalg import expm
from interferometers import L, dV_dlambdas

class Optimizer:
    def __init__(self, lr, max_steps):
        self.max_steps = max_steps
        self.opt = Adam(lr=lr)
        self.losses = []
    @staticmethod
    def mse(x, y):
        return 0.5*(x-y)**2

    

    def __call__(self, req:Requirements):
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
        except KeyboardInterrupt:
            print('exiting optimization...')
        self.losses.pop(0)
        return V


