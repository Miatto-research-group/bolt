import pytest
import strawberryfields as sf
import numpy as np
from tree import Tree
from states import State, IOSpec

def test_amplitude_onephoton():
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Fock(1) | q[0]
        sf.ops.Fock(1) | q[1]
        sf.ops.Fock(1) | q[2]
    #     sf.ops.Fock(0) | q[3]
        sf.ops.BSgate(0.7804, 0.8578)  | (q[0], q[1])
        sf.ops.BSgate(0.06406, 0.5165) | (q[2], q[3])
        sf.ops.BSgate(0.473, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.563, 0.1517)   | (q[0], q[1])
        sf.ops.BSgate(0.1323, 0.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.311, 0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4348, 0.0798)  | (q[0], q[1])
        sf.ops.BSgate(0.4368, 0.6157)  | (q[2], q[3])
        
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 4})
    results = eng.run(test_prog)
    probs = results.state.all_fock_probs()

    # getting the unitary of the interferometer
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[3:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U = S[:4, :4] + 1j*S[4:, :4]

    # |1,1,1,0> --> |1,0,1,1> or |1,1,0,1>
    s_in = State({(1,1,1,0):1.0})
    s_out1 = State({(1,0,1,1):1.0})
    s_out2 = State({(1,1,0,1):1.0})
    io1 = IOSpec(input_state=s_in, output_state=s_out1)
    io2 = IOSpec(input_state=s_in, output_state=s_out2)
    tree1 = Tree(io=io1, covariance_matrix=U)
    tree2 = Tree(io=io2, covariance_matrix=U)

    assert np.isclose(probs[1,0,1,1], abs(tree1.amplitude()[0])**2)
    assert np.isclose(probs[1,1,0,1], abs(tree2.amplitude()[0])**2)


def test_amplitude_manyphotons():
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Fock(1) | q[0]
        sf.ops.Fock(2) | q[1]
        sf.ops.Fock(1) | q[2]
    #     sf.ops.Fock(0) | q[3]
        sf.ops.BSgate(0.7804, 0.8578)  | (q[0], q[1])
        sf.ops.BSgate(0.06406, 0.5165) | (q[2], q[3])
        sf.ops.BSgate(1.473, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.563, 0.1517)   | (q[0], q[1])
        sf.ops.BSgate(0.1323, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.311, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4348, 0.0798)  | (q[0], q[1])
        sf.ops.BSgate(0.4368, 0.6157)  | (q[2], q[3])
        
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
    results = eng.run(test_prog)
    probs = results.state.all_fock_probs()

    # getting the unitary of the interferometer
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[3:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U = S[:4, :4] + 1j*S[4:, :4]

    # |1,2,1,0> --> |4,0,0,0>
    # |1,0,0,3> --> |2,2,0,0>
    s_in = State({(1,2,1,0):1.0})
    s_out1 = State({(4,0,0,0):1.0})
    s_out2 = State({(2,2,0,0):1.0})
    io1 = IOSpec(input_state=s_in, output_state=s_out1)
    io2 = IOSpec(input_state=s_in, output_state=s_out2)
    tree1 = Tree(io=io1, covariance_matrix=U)
    tree2 = Tree(io=io2, covariance_matrix=U)

    assert np.isclose(probs[4,0,0,0], abs(tree1.amplitude()[0])**2)
    assert np.isclose(probs[2,2,0,0], abs(tree2.amplitude()[0])**2)


def test_amplitude_2to1amplitudes():
    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,2,1,0] = np.sqrt(1/3)
    ket_in[1,0,0,3] = 1j*np.sqrt(2/3)
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0578)  | (q[0], q[1])
        sf.ops.BSgate(0.06406, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.473, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2517)   | (q[0], q[1])
        sf.ops.BSgate(0.3323, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.311, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4348, 0.0798)  | (q[0], q[1])
        sf.ops.BSgate(0.0368, 0.6157)  | (q[2], q[3])
        
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
    results = eng.run(test_prog)
    probs = results.state.all_fock_probs()

    # getting the unitary of the interferometer
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U = S[:4, :4] + 1j*S[4:, :4]
    # sqrt(1/3)|1,2,1,0> + 1j*sqrt(2/3)|1,0,0,3>   --->  |4,0,0,0>
    s_in = State({(1,2,1,0):np.sqrt(1/3), (1,0,0,3):1j*np.sqrt(2/3)})
    s_out = State({(4,0,0,0):1.0})
    io = IOSpec(input_state=s_in, output_state=s_out)
    tree = Tree(io = io, covariance_matrix=U)

    assert np.isclose(probs[4,0,0,0], abs(tree.amplitude()[0])**2)


def test_amplitude_2to2amplitudes():
    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,2,1,0] = np.sqrt(1/3)
    ket_in[1,0,0,3] = 1j*np.sqrt(2/3)
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0578)  | (q[0], q[1])
        sf.ops.BSgate(0.06406, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.473, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2517)   | (q[0], q[1])
        sf.ops.BSgate(0.3323, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.311, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4348, 0.0798)  | (q[0], q[1])
        sf.ops.BSgate(0.0368, 0.6157)  | (q[2], q[3])
        
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
    results = eng.run(test_prog)
    ket = results.state.ket()

    # getting the unitary of the interferometer
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U = S[:4, :4] + 1j*S[4:, :4]
    # sqrt(1/3)|1,2,1,0> + 1j*sqrt(2/3)|1,0,0,3>   --->  np.sqrt(1/7)|2,0,1,1> - 1j*np.sqrt(6/7)|4,0,0,0>
    s_in = State({(1,2,1,0):np.sqrt(1/3), (1,0,0,3):1j*np.sqrt(2/3)})
    s_out = State({(2,0,1,1):np.sqrt(1/7), (4,0,0,0):-1j*np.sqrt(6/7)})
    io = IOSpec(input_state=s_in, output_state=s_out)
    tree = Tree(io = io, covariance_matrix=U)

    assert np.isclose(np.sqrt(1/7)*ket[2,0,1,1] - 1j*np.sqrt(6/7)*ket[4,0,0,0], tree.amplitude()[0])



def test_gradients1to1():
    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,2,1,0] = 1.0
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0578)  | (q[0], q[1])
        sf.ops.BSgate(0.06406, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.473, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2517)   | (q[0], q[1])
        sf.ops.BSgate(0.3323, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.311, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4348, 0.0798)  | (q[0], q[1])
        sf.ops.BSgate(0.0368, 0.6157)  | (q[2], q[3])
        
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U1 = S[:4, :4] + 1j*S[4:, :4]

    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,2,1,0] = 1.0
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0568)  | (q[0], q[1])
        sf.ops.BSgate(0.06306, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.472, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2507)   | (q[0], q[1])
        sf.ops.BSgate(0.3333, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.312, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4358, 0.0788)  | (q[0], q[1])
        sf.ops.BSgate(0.0358, 0.6157)  | (q[2], q[3])
        
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U2 = S[:4, :4] + 1j*S[4:, :4]
    dU = U2-U1

    s_in = State({(1,2,1,0):1.0})
    s_out = State({(1,0,0,3):1.0})
    io = IOSpec(input_state=s_in, output_state=s_out)
    tree1 = Tree(io = io, covariance_matrix=U1)
    tree2 = Tree(io = io, covariance_matrix=U2)

    amp1,grad1 = tree1.amplitude()
    amp2,grad2 = tree2.amplitude()
    print(f'tree builds input: {io.building_input}')
    print(grad1)
    print(amp2, amp1 + np.sum(grad1*dU))
    print(amp2-amp1, np.sum(grad1*dU))
    assert np.isclose(amp2, amp1 + np.sum(grad1*dU))
    assert np.isclose(amp1, amp2 - np.sum(grad2*dU))


def test_gradients_onephoton():
    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,1,1,0] = 1.0
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0578)  | (q[0], q[1])
        sf.ops.BSgate(0.06406, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.473, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2517)   | (q[0], q[1])
        sf.ops.BSgate(0.3323, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.311, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4348, 0.0798)  | (q[0], q[1])
        sf.ops.BSgate(0.0368, 0.6157)  | (q[2], q[3])
        
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U1 = S[:4, :4] + 1j*S[4:, :4]

    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,2,1,0] = 1.0
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0568)  | (q[0], q[1])
        sf.ops.BSgate(0.06306, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.472, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2507)   | (q[0], q[1])
        sf.ops.BSgate(0.3333, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.312, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4358, 0.0788)  | (q[0], q[1])
        sf.ops.BSgate(0.0358, 0.6157)  | (q[2], q[3])
        
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U2 = S[:4, :4] + 1j*S[4:, :4]
    dU = U2-U1

    s_in = State({(1,1,1,0):1.0})
    s_out = State({(1,0,1,1):1.0})
    io = IOSpec(input_state=s_in, output_state=s_out)
    tree1 = Tree(io = io, covariance_matrix=U1)
    tree2 = Tree(io = io, covariance_matrix=U1+dU)

    amp1,grad1 = tree1.amplitude()
    amp2,grad2 = tree2.amplitude()
    print(f'tree builds input: {io.building_input}')
    print(grad1)
    print(amp2, amp1 + np.sum(grad1*dU))
    print(amp2-amp1, np.sum(grad1*dU))
    assert np.isclose(amp2 - amp1, np.sum(grad1*dU))
    assert np.isclose(amp1 - amp2, -np.sum(grad2*dU))

def test_gradients2to1():
    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,2,1,0] = np.sqrt(1/3)
    ket_in[1,0,0,3] = 1j*np.sqrt(2/3)
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0578)  | (q[0], q[1])
        sf.ops.BSgate(0.06406, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.473, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2517)   | (q[0], q[1])
        sf.ops.BSgate(0.3323, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.311, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4348, 0.0798)  | (q[0], q[1])
        sf.ops.BSgate(0.0368, 0.6157)  | (q[2], q[3])
        
    # eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
    # ket_out_1 = eng.run(test_prog).state.ket()
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U1 = S[:4, :4] + 1j*S[4:, :4]

    ket_in = np.zeros((5,5,5,5), dtype=np.complex128)
    ket_in[1,2,1,0] = np.sqrt(1/3)
    ket_in[1,0,0,3] = 1j*np.sqrt(2/3)
    test_prog = sf.Program(4) # 4 modes
    with test_prog.context as q:
        sf.ops.Ket(ket_in) | (q[0], q[1], q[2], q[3])
        sf.ops.BSgate(0.1804, 0.0568)  | (q[0], q[1])
        sf.ops.BSgate(0.06306, 1.5165) | (q[2], q[3])
        sf.ops.BSgate(1.462, 0.1176)   | (q[1], q[2])
        sf.ops.BSgate(0.263, -0.2507)   | (q[0], q[1])
        sf.ops.BSgate(0.3333, 1.9946)  | (q[2], q[3])
        sf.ops.BSgate(0.312, -0.3231)   | (q[1], q[2])
        sf.ops.BSgate(0.4358, 0.0788)  | (q[0], q[1])
        sf.ops.BSgate(0.0358, 0.6157)  | (q[2], q[3])
        
    # eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 5})
    # ket_out_2 = eng.run(test_prog).state.ket()
    prog_unitary = sf.Program(4)
    prog_unitary.circuit = test_prog.circuit[1:]
    prog_compiled = prog_unitary.compile(compiler="gaussian_unitary")
    S = prog_compiled.circuit[0].op.p[0]
    U2 = S[:4, :4] + 1j*S[4:, :4]
    dU = U2-U1

    s_in = State({(1,2,1,0):np.sqrt(1/3), (1,0,0,3):1j*np.sqrt(2/3)})
    s_out = State({(4,0,0,0):1.0})
    io = IOSpec(input_state=s_in, output_state=s_out)
    tree1 = Tree(io = io, covariance_matrix=U1)
    tree2 = Tree(io = io, covariance_matrix=U1+dU)

    amp1,grad1 = tree1.amplitude()
    amp2,grad2 = tree2.amplitude()
    print(amp2, amp1 + np.sum(grad1*dU))
    print(np.sum(grad1*dU), amp2-amp1)
    assert np.isclose(amp2, amp1 + np.sum(grad1*dU))
    assert np.isclose(amp1, amp2 - np.sum(grad2*dU))