# interferometers

This is a library that allows one to simulate and optimize interferometers at the quantum level. 

# How to use

### 1. Create input-output states
```python
from states import State, IOSpec

_in = State({(1,1,1,0,0,0):1.0}) # |1,1,1,0,0,0>
_out = State({(1,0,1,0,1,0):1.0}) # |1,0,1,0,1,0>

# IOSpec is an input-output relation (pure state in -> pure state out)
io = IOSpec(input_state = _in, output_state = _out)
```

### 2. Create a `Requirements` object for multiple input-output relations
```python
from states import Requirements

# format: {IOSpec:required probability, etc...}
req = Requirements({io:1.0})
```

### 3. Find interferometer that best satisfies the requirement
```python
from optimizer import Optimizer

opt = Optimizer(lr = 0.01, max_steps=200)

cov_matrix = opt(req)
```

