import time
import numpy as np
from numpy_ml.neural_nets.optimizers import Adam
from tqdm import trange
from multiprocessing import Pool

# from scipy.linalg import expm
from typing import Tuple, List
from .expm import expm
from .tree import Tree
from .liealgebra import L, dV_dlambdas
from .states import State,IOSpec,Requirements
from .utils import partition,pos,all_outputs

class General_State_Discriminator:
    """
    Interferometer optimizer class. Create an instance by specifying:
        lr (float): learning rate
        epsilon (float): the optimization keeps going until the drop in loss is larger than epsilon
        max_steps: (int): stop at max_steps if the epsilon criterion is not met yet

        Usage:
        >>> opt = Optimizer(lr=0.01, epsilon=1e-4)
        >>> cov_matrix = opt(requirements)
    """
    def __init__(self, lr:float, epsilon:float = 1e-6, max_steps:int = 1000, cov_matrix_init=None,com:bool=False):#, natural:bool = False):
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.losses = []
        self.elapsed = 0.0
        self.cov_matrix_init = cov_matrix_init
        self.op=[]
        self.op_g=[]
        self.states=[]
        self.lr = lr
        self.com=com

    @staticmethod
    def discrim_states(n:int,l:int): #n=no of photons, l=no of modes
        b=[]
        for i in range(n):
            i+=1
            w=partition(i,(1,)*l)
            for j in w:
                b.append(j)
        c=partition(n,(n,)*l)
        a=[]
        for i in b:
            d=[]
            d.append(i)
            for j in c:
                if pos(i)==pos(j):
                    d.append(j)
            a.append(d)
        return a
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

    def gra(self,p:[int]):
        i=p[0]
        j=p[1]
        i1=p[2]
        i2=p[3]
        r=p[4]
        a=self.states[r][i1]
        b=self.states[r][i2]
        a1 = self.op[i].get(a)
        da1_dV = self.op_g[i].get(a)
        a2 = self.op[j].get(b)
        da2_dV = self.op_g[j].get(b)
        p1 = abs(a1)**2
        p2 = abs(a2)**2
        loss = p1*p2
        dL_da1=np.conj(a1)*p2
        dL_da2=np.conj(a2)*p1
        da1_dA = np.block([[da1_dV, -1j*da1_dV],[1j*da1_dV, da1_dV]])
        da2_dA = np.block([[da2_dV, -1j*da2_dV],[1j*da2_dV, da2_dV]])
        grad_update = 2*np.real((dL_da1 * da1_dA)+(dL_da2 * da2_dA))
        return (loss,grad_update)

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
        self.states=self.discrim_states(n=num_photons,l=num_modes)
        tic = time.time()
        
        if self.com:
            V = self.cov_matrix_init
        else:
            lambdas = np.random.normal(size=size, scale=0.01)
            V = expm(L(lambdas))
        
        A = np.block([[np.real(V), -np.imag(V)],[np.imag(V), np.real(V)]])
        pool=Pool()
        self.losses = [1e6]
        try:
            progress = trange(self.max_steps)
            for step in progress:
                grad_update = 0
                loss = 0
                self.op=[]
                self.op_g=[]
                #tic1 = time.time()
                for i in range(lengthst):
                    u,v= all_outputs(st[i], V, grad=True)
                    # op is an array of dictionaries, each element being the corresponding output state for each input state 
                    self.op.append(u)
                    self.op_g.append(v)
                #toc1 = time.time()
                #print("Time to store all outputs is ",toc1-tic1)
                w=[]
                for i in range(lengthst):
                    for j in range(lengthst):
                        if (j>i):
                            for r in range(len(self.states)):
                                len1=len(self.states[r]) 
                                for i1 in range(1,len1):
                                    for i2 in range(1,len1):
                                        w.append([i,j,i1,i2,r])

                
                grad_update2=pool.map(self.gra,w)
                
                for inu in grad_update2:
                    loss+=inu[0]
                    grad_update+=inu[1]
                #print(grad_update)
                self.losses.append(loss)
                Q = grad_update
                D = 0.5*(Q - A @ Q.T @ A) # natural gradient
                A = A @ expm(self.lr * D.T @ A)
                V = A[:len(V), :len(V)] + 1j*A[len(V):, :len(V)]
                
                progress.set_description(f"{step}:loss = {loss:.4f}")
                if abs(self.losses[-2] - self.losses[-1]) < self.epsilon:
                    break
        except KeyboardInterrupt:
            print('gracefully stopping optimization...')
        self.losses.pop(0)
        toc = time.time()
        self.elapsed = toc-tic
        pool.close()
        pool.join()
        return V