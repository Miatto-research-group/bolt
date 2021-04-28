### What is `bolt`?

`bolt` is a \*very fast\* library that allows one to simulate and optimize interferometers at the quantum level. 

### How can it be so fast?
`bolt` does its magic by computing only the input-output amplitudes of the interferometer that are needed, rather than computing the entire transformation tensor up to a given Fock space cutoff `N` (i.e. *all* of the amplitudes). 

Then, it performs the gradient optimization in the Lie algebra of the unitary group, which allows it to update the covariance matrix directly, without worrying about decomposing the interferometer into some arrangement of beam splitters and phase shifters.

It also (optionally) implements the natural gradient in the Orthogonal group, for an even faster convergence.

### How to use

#### 1. Create input and output states
The `State` class is a dictionary of ket:amplitude pairs:
```python
from bolt import State, IOSpec

_in = State({(1,1,1,0,0,0):1.0}) # |1,1,1,0,0,0>
_out = State({(1,0,1,0,1,0):1.0}) # |1,0,1,0,1,0>

# IOSpec is an input-output relation (pure state in -> pure state out)
io = IOSpec(input_state = _in, output_state = _out)
```

#### 2. Create a `Requirements` object for multiple input-output relations
The `Requirements` class collects all of the required input-output relations that we require from the interferometer.
Generally, an interferometer that satisfies all of the required relations does not exist, but the optimizer will try to find one
that satisfies them with the highest probability.
```python
from bolt import Requirements

# format: {IOSpec:weight, etc...}
req = Requirements({io:1.0})
```

#### 3. Find the interferometer that best satisfies the requirements
Note that the *first time* the optimizer is called, the various `numba` functions in the code are compiled.
Subsequent calls will start immediately, until you restart the ipython kernel.
```python
from bolt import Optimizer
opt = Optimizer(lr = 0.01, max_steps=500)

cov_matrix = opt(req)
print(f'The search took {opt.elapsed:.3f} seconds')

import matplotlib.pyplot as plt
plt.plot(opt.losses)
```

### Did you blink?
Let's increase the complexity (16 modes, 12 photons). It should still be reasonably fast (44 it/s on my laptop):
```python
from bolt import State, IOSpec, Requirements, Optimizer

_in = State({(1,1,3,1,2,0,0,0,0,0,0,3,0,0,1,0):1.0}) # |1,1,3,1,2,0,0,0,0,0,0,3,0,0,1,0>
_out = State({(1,0,1,0,2,0,1,2,1,0,0,2,0,1,1,0):1.0}) # |1,0,1,0,2,0,1,2,1,0,0,2,0,1,1,0>

io = IOSpec(input_state = _in, output_state = _out)
req = Requirements({io:1.0})

opt = Optimizer(lr = 0.02)
cov_matrix = opt(req)
```

### All possible output amplitudes
`Bolt` can generate the complete output state as well:
```python
from bolt.utils import all_outputs
from scipy.stats import unitary_group
from bolt import State

state_in = State({(0,8,0,8):np.sqrt(1/4), (8,0,8,0):np.sqrt(1/4), (8,0,0,8):np.sqrt(1/4), (0,8,8,0):np.sqrt(1/4)})
V = unitary_group.rvs(state_in.num_modes) # interferometer unitary

out,_ = all_outputs(state_in, V, grad=False)
```

### Fun Experiments
Note that at times the optimizer gets stuck in a local minimum. Run the optimization a few times to assess how often this happens.

#### Bell state analyzer
States from Eq. (2) in [PRA 94, 042331 (2011)](https://pdfs.semanticscholar.org/392a/3f99eb07c919da782831939082fa4eaac802.pdf).
```python
import numpy as np
from bolt import State, IOSpec, Requirements, Optimizer

psip = State({(1,0,0,1):np.sqrt(1/2), (0,1,1,0):np.sqrt(1/2)})
psim = State({(1,0,0,1):np.sqrt(1/2), (0,1,1,0):-np.sqrt(1/2)}) 
phip = State({(1,0,1,0):np.sqrt(1/2), (0,1,0,1):np.sqrt(1/2)}) 
phim = State({(1,0,1,0):np.sqrt(1/2), (0,1,0,1):-np.sqrt(1/2)})

io1 = IOSpec(psip, State({(1,1,0,0):np.sqrt(1/2), (0,0,1,1):np.sqrt(1/2)}))
io2 = IOSpec(psim, State({(1,0,0,1):np.sqrt(1/2), (0,1,1,0):-np.sqrt(1/2)}))
io3 = IOSpec(phip, State({(2,0,0,0):1/2, (0,2,0,0):1/2, (0,0,2,0):1/2, (0,0,0,2):1/2}))
io4 = IOSpec(phim, State({(2,0,0,0):1/2, (0,0,2,0):1/2, (0,2,0,0):-1/2, (0,0,0,2):-1/2}))

req = Requirements({io1:1.0, io2:1.0, io3:1.0, io4:1.0})
opt = Optimizer(lr = 0.01)
cov_matrix = opt(req)
print(f'The search took {opt.elapsed:.3f} seconds')

import matplotlib.pyplot as plt
plt.plot(opt.losses);
```

#### GHZ state generation
Here we find out that we can generate a GHZ state with probability 1/2.
```python
import numpy as np
import matplotlib.pyplot as plt
from bolt import State, IOSpec, Requirements, Optimizer

in_111 = State({(1,1,1,0,0,0):1.0}) 

out_GHZ = State({(1,0,1,0,1,0):np.sqrt(1/2), (0,1,0,1,0,1):np.sqrt(1/2)}) 


io = IOSpec(in_111, out_GHZ)
req = Requirements({io:1.0})
opt = Optimizer(lr = 0.01)
cov_matrix = opt(req)

print(f'The search took {opt.elapsed:.3f} seconds')
plt.plot(opt.losses);
```

#### GHZ state generation with natural gradient
Let's push the limits with 11 photons and 22 modes, while using the natural gradient.
Compare with the Lie Algebra implementation and notice the difference.
Note that the natural gradient may be a bit sensible to the learning rate.
```python
import numpy as np
import matplotlib.pyplot as plt
from bolt import State, IOSpec, Requirements, Optimizer

p = 11
in_ = State({(1,)*p + (0,)*p:1.0}) 

out_GHZ = State({(1,0)*p:np.sqrt(1/2), (0,1)*p:np.sqrt(1/2)}) 


io = IOSpec(in_, out_GHZ)
req = Requirements({io:1.0})
opt = Optimizer(lr = 0.02, natural=True) # change to natural=False for Lie Algebra implementation
cov_matrix = opt(req)

print(f'The search took {opt.elapsed:.3f} seconds')
plt.plot(opt.losses[1:]);
```