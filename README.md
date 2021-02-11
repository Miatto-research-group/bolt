# interferometers

`interferometers` is a \*very fast\* library that allows one to simulate and optimize interferometers at the quantum level. 

## How to use

### 1. Create input and output states
The `State` class is a dictionary of ket:amplitude pairs:
```python
from interferometers import State, IOSpec

_in = State({(1,1,1,0,0,0):1.0}) # |1,1,1,0,0,0>
_out = State({(1,0,1,0,1,0):1.0}) # |1,0,1,0,1,0>

# IOSpec is an input-output relation (pure state in -> pure state out)
io = IOSpec(input_state = _in, output_state = _out)
```

### 2. Create a `Requirements` object for multiple input-output relations
The `Requirements` class collects all of the required input-output relations that we require from the interferometer.
Generally, an interferometer that satisfies all of the required relations does not exist, but the optimizer will try to find one
that satisfies them the best.
```python
from interferometers import Requirements

# format: {IOSpec:weight, etc...}
req = Requirements({io:1.0})
```

### 3. Find the interferometer that best satisfies the requirements
Note that the *first time* the optimizer is called, the various `numba` functions in the code are compiled.
Subsequent calls will start immediately, until you restart the ipython kernel.
```python
from interferometers import Optimizer
opt = Optimizer(lr = 0.01)

cov_matrix = opt(req)
print(f'The search took {opt.elapsed:.3f} seconds')

import matplotlib.pyplot as plt
plt.plot(opt.losses)
```

## Did you blink?
Let's increase the complexity (16 modes, 12 photons). It should still be reasonably fast (44 it/s on my laptop):
```python
from interferometers import State, IOSpec, Requirements, Optimizer

_in = State({(1,1,3,1,2,0,0,0,0,0,0,3,0,0,1,0):1.0}) # |1,1,3,1,2,0,0,0,0,0,0,3,0,0,1,0>
_out = State({(1,0,1,0,2,0,1,2,1,0,0,2,0,1,1,0):1.0}) # |1,0,1,0,2,0,1,2,1,0,0,2,0,1,1,0>

io = IOSpec(input_state = _in, output_state = _out)
req = Requirements({io:1.0})

opt = Optimizer(lr = 0.02)
cov_matrix = opt(req)
```

## How is it that fast?
`interferometers` does its magic by computing only the input-output amplitudes of the interferometer that are needed, rather than computing *all* of the amplitudes up to a given cutoff. Then, it performs the gradient optimization in the Lie algebra of the unitary group, which allows it to update the covariance matrix directly, without worrying about decomposing the interferometer into some arrangement of beam splitters and phase shifters.