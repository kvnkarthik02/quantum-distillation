from qutip import basis, sigmax, sigmay, sigmaz, Qobj, fidelity
import numpy as np
# Initial state (|0⟩)
rho_initial = basis(2, 0) * basis(2, 0).dag()
# Noise probability
p = 0.9  #adjust this value 0-1 for different fidelity values. 
# Define the Kraus operators
K0 = np.sqrt(1 - p) * Qobj(np.eye(2))
K1 = np.sqrt(p / 3) * sigmax()
K2 = np.sqrt(p / 3) * sigmay()
K3 = np.sqrt(p / 3) * sigmaz()
kraus_operators = [K0, K1, K2, K3]
# Apply the Kraus operators to the density matrix
noisy_state = sum(K * rho_initial * K.dag() for K in kraus_operators)
fidelity = fidelity(noisy_state, rho_initial)
print("Fidelity: ", fidelity)
print("Noisy state:")
print(noisy_state)
