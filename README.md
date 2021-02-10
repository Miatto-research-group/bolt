# interferometers

This is a library that allows one to simulate and optimize interferometers at the quantum level. 

# How to use

### 1. Create input and output states
The `State` class is a dictionary of ket:amplitude pairs:
```python
from states import State, IOSpec

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
from states import Requirements

# format: {IOSpec:required probability, etc...}
req = Requirements({io:1.0})
```

### 3. Find interferometer that best satisfies the requirements
Note that the first time the optimizer is called, the various `numba` functions in the code are compiled
```python
from optimizer import Optimizer
import matplotlib.pyplot as plt

opt = Optimizer(lr = 0.01)

cov_matrix = opt(req)
print(f'The sarch took {opt.elapsed:.3f} seconds')
plt.plot(opt.losses)
```
