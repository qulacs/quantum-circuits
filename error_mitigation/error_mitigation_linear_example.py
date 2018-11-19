"""
error mitigation with extrapolation
"""

from qulacs import Observable
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import Probabilistic, X, Y, Z

import matplotlib.pyplot as plt
from numpy import polyfit


def error_mitigation_extrapolate_linear(quantum_circuit_list, error_list, initial_state, 
                                        obs, n_circuit_sample = 1000, return_full = False):
    """error_mitigation_extrapolate_linear

    Args:
        quantum_circuit_list (:class:`list`):
            list of quantum circuit with deifferent error rate
        initial_state (:class:`qulacs.QuantumState`):
            list of quantum circuit with deifferent error rat
        error_list (:class:`list`):
            list of error rate for each quantum circuit
        obs (:class:`qulacs.Observable`):
            measured observable
    Returns:
        :class:`float` when return full is False (default)
        :class:`tuple` when return full is True
    """
    exp_array = []
    for quantum_circuit in quantum_circuit_list:
        exp = 0
        for _ in [0]*n_circuit_sample:
            state = initial_state.copy()
            quantum_circuit.update_quantum_state(state)
            exp += obs.get_expectation_value(state)
        exp_array.append(exp/n_circuit_sample)

    fit_coefs = polyfit(error_list, exp_array, 1)
    if return_full:
        return fit_coefs[1], exp_array, fit_coefs
    else:
        return fit_coefs[1]


def main():
    import numpy as np
    n_qubit = 2
    obs = Observable(n_qubit)
    initial_state = QuantumState(n_qubit)
    obs.add_operator(1, "Z 0 Z 1")
    circuit_list = []
    p_list = [0.02, 0.04, 0.06, 0.08]
    
    #prepare circuit list
    for p in p_list:
        circuit = QuantumCircuit(n_qubit)
        circuit.add_H_gate(0)
        circuit.add_RY_gate(1, np.pi/6)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_gate(Probabilistic([p/4, p/4, p/4], [X(0), Y(0), Z(0)])) #depolarizing noise
        circuit.add_gate(Probabilistic([p/4, p/4, p/4], [X(1), Y(1), Z(1)])) #depolarizing noise
        circuit_list.append(circuit)

    #get mitigated output
    mitigated, non_mitigated_array, fit_coefs = error_mitigation_extrapolate_linear(circuit_list, p_list, initial_state, obs, n_circuit_sample = 100000, return_full = True)
    
    #plot the result
    p = np.linspace(0, max(p_list), 100)
    plt.plot(p, fit_coefs[0]*p+fit_coefs[1], linestyle = "--", label = "linear fit")
    plt.scatter(p_list, non_mitigated_array, label = "un-mitigated")
    plt.scatter(0, mitigated, label = "mitigated output")

    #prepare the clean result
    state = QuantumState(n_qubit)
    circuit = QuantumCircuit(n_qubit)
    circuit.add_H_gate(0)
    circuit.add_RY_gate(1, np.pi/6)
    circuit.add_CNOT_gate(0, 1)
    circuit.update_quantum_state(state)
    plt.scatter(0, obs.get_expectation_value(state), label = "True output")
    plt.xlabel("error rate")
    plt.ylabel("expectation value")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

