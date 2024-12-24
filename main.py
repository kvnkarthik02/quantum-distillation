from qutip import basis, sigmax, sigmay, sigmaz, Qobj, fidelity, tensor, sigmax, sigmay
import numpy as np
# Initial state (|0⟩)
psi_singlet = (basis(2, 0) * basis(2, 1).dag() - basis(2, 1) * basis(2, 0).dag()).unit()
print("type " + psi_singlet.type )

# Noise probability
p = 1  #adjust this value 0-1 for different fidelity values. 

# Define the Kraus operators
K0 = np.sqrt(1 - p) * Qobj(np.eye(2))
K1 = np.sqrt(p / 3) * sigmax()
K2 = np.sqrt(p / 3) * sigmay()
K3 = np.sqrt(p / 3) * sigmaz()
kraus_operators = [K0, K1, K2, K3]

# Apply the Kraus operators to the density matrix
noisy_state = sum(K * psi_singlet * K.dag() for K in kraus_operators)
initial_fidelity = fidelity(noisy_state, psi_singlet)
print("--------------------------------")
print(" Initial Fidelity: ", initial_fidelity)
print("Noisy state:")
print(noisy_state)
print("--------------------------------")

#start purification protocol
# Step 1: Apply unilateral π-rotation (Pauli-Y)
pauli_y = sigmay()
rotated_state = pauli_y * noisy_state * pauli_y.dag()
# Step 2: Apply BXOR operation
measurement_outcome = np.random.choice([0, 1], p=[0.5, 0.5])
if measurement_outcome == 1:
        # Apply Pauli-X on both qubits
        # result_state = tensor(sigmax(), sigmax()) * rotated_state * tensor(sigmax(), sigmax()).dag()
        result_state = sigmax() * rotated_state * sigmax().dag()
else:
    result_state = rotated_state


# Fidelity after purification
final_fidelity = fidelity(result_state, psi_singlet)
print("--------------------------------")
print(" Final Fidelity: ", final_fidelity)
print("Noisy state:")
print(noisy_state)
print("--------------------------------")   

