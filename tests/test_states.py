import numpy as np
import pytest
from states import State, IOSpec

def test_state_normalization():
    state = State()
    state[(1,2,3)] = 0.5
    state[(1,3,9)] = 0.5
    state.normalize()
    assert np.isclose(sum(abs(s)**2 for s in state.values()), 1)

def test_requirement_1to1():
    req = IOSpec(State({(1,2):1.0}), State({(0,3):1.0}))
    gen = (i for i in req.paths)
    assert next(gen) == ((1,2),(0,3),1.0)
    with pytest.raises(StopIteration):
        next(gen)

def test_requirement_2to1():
    req = IOSpec(State({(1,2):1.0, (0,3):0.5}), State({(2,1):1.0}))
    gen = (i for i in req.paths)
    assert next(gen) == ((2,1),(1,2), 1.0)
    assert next(gen) == ((2,1),(0,3), 0.5)
    with pytest.raises(StopIteration):
        next(gen)

def test_requirement_1to2():
    req = IOSpec(State({(1,2):1.0}), State({(0,3):0.5, (2,1):1.0}))
    gen = (i for i in req.paths)
    assert next(gen) == ((1,2),(0,3), 0.5)
    assert next(gen) == ((1,2),(2,1), 1.0)
    with pytest.raises(StopIteration):
        next(gen)

def test_requirement_2to2():
    req = IOSpec(State({(1,2):1.0, (0,3):0.4}), State({(3,0):0.5, (2,1):1.0}))
    gen = (i for i in req.paths)
    assert next(gen) == ((3,0),(1,2),0.5)
    assert next(gen) == ((3,0),(0,3),0.2)
    assert next(gen) == ((2,1),(1,2),1.0)
    assert next(gen) == ((2,1),(0,3),0.4)
    with pytest.raises(StopIteration):
        next(gen)