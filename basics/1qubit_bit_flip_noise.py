"""
probabilistic bit flip noise
"""

from qulacs import Observable
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import Probabilistic, X

import matplotlib.pyplot as plt

obs = Observable(1)
obs.add_operator(1, "Z 0")
state = QuantumState(1)
circuit = QuantumCircuit(1)
p = 0.1 # probability of bit flip
n_circuit_sample = 10000
n_depth = 20 # the number of probabilistic gate

probabilistic_pauli_gate = Probabilistic([p],[X(0)]) #define probabilistic gate

circuit.add_gate(probabilistic_pauli_gate) # add the prob. gate to the circuit

exp_array = []
for depth in range(n_depth):
    exp = 0
    for i in [0]*n_circuit_sample:
        state.set_zero_state()
        for _ in range(depth):
            circuit.update_quantum_state(state) # apply the prob. gate
        exp += obs.get_expectation_value(state) # get expectation value for one sample of circuit 
    exp /= n_circuit_sample # get overall average
    exp_array.append(exp)

#plot
plt.plot(range(n_depth), exp_array)
plt.show()