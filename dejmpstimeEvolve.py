#imports
from qutip import *
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, cnot
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from qutip.qip.operations.gates import (rx, ry, rz)
import random
from qutip.measurement import measure, measure_povm

phi_plus = bell_state("00")
phi_minus = bell_state("01")
psi_plus = bell_state("10")
psi_minus = bell_state("11")

def bds_state(F):
    p1 = F
    p2 = (1-p1)/2
    p3 = (1-p1)/3
    p4 = (1-p1)/6

    state = p1 * psi_minus * psi_minus.dag() + p2 * (psi_plus * psi_plus.dag()) + p3 * phi_minus * phi_minus.dag() + p4 * (phi_plus * phi_plus.dag())
    return state

def apply_twirling(rho):
    """
    Apply twirling operation to convert a general two-qubit state into WBDSerner state
    using the transformations from Bennett protocol.

    Parameters:
        rho (Qobj): Input two-qubit state

    Returns:
        Qobj: BDS state after twirling  
    """
    # Define Pauli matrices and identity
    I2 = qeye(2)
    sx = sigmax()
    sy = sigmay()
    # sz = sigmaz()

    # Define K transformations from Bennett protocol
    u1 = (I2 + 1j * sx) / np.sqrt(2)
    u2 = (I2 - 1j * sy) / np.sqrt(2)
    u3 = (1j * basis([2], 0) * basis([2], 0).dag()) + basis([2], 1) * basis([2], 1).dag()
    u4 = I2

    K = []
    for u in [u1, u2, u3, u4]:
        K.append(tensor(u, u))

    # Apply twirling operation
    bracket_term = Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
    rho_w = Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
    for i in range(4):
        term = K[i].dag() @ K[i].dag() @ rho @ K[i] @ K[i]
        bracket_term += term
    # for j in range(3):
    #     rho_w += K[j].dag() @ bracket_term @ K[j]
    bracket_term = bracket_term / 4

    return bracket_term

def calc_fidelity(state, p0 = bell_state('11')):
    try:
        return (fidelity(state, p0))
    except:
        return (fidelity(state, tensor(p0, p0)))

def rx_rotation(state):
    rx_rotation_0 = tensor(rx(pi/2), qeye(2), qeye(2), qeye(2))
    rx_rotation_1 = tensor(qeye(2), rx(-pi/2), qeye(2), qeye(2))
    rx_rotation_2 = tensor(qeye(2), qeye(2), rx(pi/2), qeye(2))
    rx_rotation_3 = tensor(qeye(2), qeye(2), qeye(2), rx(-pi/2))

    state = rx_rotation_0 * state * rx_rotation_0.dag()
    state = rx_rotation_1 * state * rx_rotation_1.dag()
    state = rx_rotation_2 * state * rx_rotation_2.dag()
    state = rx_rotation_3 * state * rx_rotation_3.dag()
    return state


def unilateral_y_rotation(state):
    #create custom unilateral y-rotation operators
    u_rotation_qubit0 = tensor(sigmay(), qeye(2), qeye(2), qeye(2))
    u_rotation_qubit2 = tensor(qeye(2), qeye(2), sigmay(), qeye(2))

    #apply unilateral y-rotation on qubit 0 and 2 
    state = u_rotation_qubit0 * state * u_rotation_qubit0.dag()
    state = u_rotation_qubit2 * state * u_rotation_qubit2.dag()
    return state

def bilateral_xor(rho):
    #define zero and one basis for our 4-qubit system
    zero_state = basis(2,0) * basis(2,0).dag()
    one_state = basis(2,1) * basis(2,1).dag()

    # generate bxor operators for different operators
    bxor02 = tensor(zero_state, qeye(2), qeye(2), qeye(2)) + tensor(one_state, qeye(2), sigmax(), qeye(2))
    bxor13 = tensor(qeye(2), zero_state, qeye(2), qeye(2)) + tensor(qeye(2), one_state, qeye(2), sigmax())

    #generate bxor operators for different operators
    # bxor02 = tensor(zero_state, qeye(2), qeye(2), qeye(2)) + tensor(one_state, qeye(2), sigmax(), qeye(2))
    # bxor13 = tensor(qeye(2), zero_state, qeye(2), qeye(2)) + tensor(qeye(2), one_state, sigmax(), qeye(2))

    rho = bxor02 * rho * bxor02.dag()
    rho = bxor13 * rho * bxor13.dag()
    return rho

def measure_qubits(state):
    Z0 = ket2dm(basis(2, 0))
    Z1 = ket2dm(basis(2, 1))
    # The measurement POVM elements act as identity on qubits 0 and 1:
    PZ = [
        tensor(qeye(2), qeye(2), Z0, Z0),
        tensor(qeye(2), qeye(2), Z1, Z1),
        tensor(qeye(2), qeye(2), Z0, Z1),
        tensor(qeye(2), qeye(2), Z1, Z0)
    ]
    
    outcome, final = measure_povm(state, PZ)
    return outcome, final

def generate_collapse_operators(T1, T2):
    gam = 1 / T1  # probability of a type 1 error

    dep = 1 / T2  # probability of a type 2 error

    # operators acting on A whilst B is in flight
    c_ops_partial = [np.sqrt(gam) * tensor(destroy(2), qeye(2)),
                     np.sqrt(dep) * tensor(sigmaz(), qeye(2))]

    # operators acting on both A and B once B has arrived
    c_ops_full = c_ops_partial + [np.sqrt(gam) * tensor(qeye(2), destroy(2)),
                                  np.sqrt(dep) * tensor(qeye(2), sigmaz())]

    return c_ops_partial, c_ops_full

def apply_locc_noise(state, waiting_time, T1, T2):
    """
    Apply additional noise during the classical communication waiting time
    via a Lindblad evolution.

    Parameters:
        state (Qobj): The 2-qubit state (after purification) to be further evolved.
        waiting_time (float): Waiting time (seconds) during which the state decoheres.
        T1 (float): Relaxation time.
        T2 (float): Dephasing time.

    Returns:
        Qobj: The state after waiting noise.
    """
    # Here we assume both qubits are in memory so use the full collapse operators.
    _, c_ops_full = generate_collapse_operators(T1, T2)
    H = tensor(qeye(2), qeye(2))  # trivial Hamiltonian evolution
    t_list = np.linspace(0, waiting_time, 1000)
    result = mesolve(H, state, t_list, c_ops_full)
    return result.states[-1]



def dejmps(initial_state, threshold_fidelity, T1, T2, delay_time, max_iterations=10):
    current_state = initial_state.copy()
    print()
    fidelity_history = [0.7]    # print("Final Fidelity:", input_fidelity)

    for iteration in range(max_iterations):
        rho_initial = tensor(current_state, current_state)
        success = False
        
        while not success:

            #START PURIFICATION PROCESS``
            print("------------------------------------------------------------------")        
            print(f"Iteration {iteration + 1}")
            rho = rho_initial.copy()
            
            print("Fidelity before purification: ", calc_fidelity(rho))

            #apply x-rotations on each qubit alternating b/w pi/2 and (-pi/2)
            rho = rx_rotation(rho)
            
            #apply bilateral xor on qubits (0,2) and (1,3)
            rho = bilateral_xor(rho)

            #measure target qubits 2 and 3, keep 
            measurement_result, rho_post =  measure_qubits(rho)
            
            #check for successful purification
            if measurement_result in [0,1]:
                success = True
                purified_state = rho_post.ptrace([0,1])

                y_rotation = tensor(sigmay(), qeye(2))
                purified_state = y_rotation * purified_state * y_rotation.dag()

                print("Fidelity after purification before LOCC noise: ", np.square(fidelity(purified_state, bell_state('11'))))
                purified_state_after_wait = apply_locc_noise(purified_state, delay_time, T1, T2)
                fidelity_val = np.square(fidelity(purified_state_after_wait, bell_state('11')))
                fidelity_history.append(fidelity_val)
                print(f"Fidelity after purification and LOCC noise: {fidelity_val}")

                print("------------------------------------------------------------------")
                if fidelity_val > threshold_fidelity:
                    print('SUCCESSFUL')
                    return purified_state_after_wait, fidelity_history
                else:
                    # current_state = purified_state_after_wait
                    current_state = apply_twirling(purified_state_after_wait)
            else:
                # print("------------------------------------------------------------------")
                print("Purification failed. Retrying...")
                # print("------------------------------------------------------------------")
    
    print(f"Max iterations ({max_iterations}) reached. Final fidelity: {fidelity_history[-1] if fidelity_history else 'N/A'}")
    return current_state, fidelity_history


# channel_lengths = np.linspace(10, 90, 9, endpoint=True)
# delays = [0.001]
# bar_gr_result_f = {delay: [] for delay in delays}
# quantum_channel_lengths = [20, 22]
# memory_params = {"T1": [1.14], "T2": [0.5]}
# # memory_params = {"T1": [0.0256], "T2": [0.034]}
# # memory_params = {"T1": [200], "T2": [0.5]}
# initial_state = bds_state(0.7)

# purified_state, fidelity_history = dejmps(
#             initial_state=initial_state,
#             threshold_fidelity=0.999999999999999999999999999999,
#             T1=memory_params["T1"][0],
#             T2=memory_params["T2"][0],
#             delay_time=1e-4,
#             max_iterations=50
#         )

#--------------------------------------------------------------------------------------------------
# # CODE FOR CREATING MAX FIDELITY V LATENCY PLOT. SET MAX ITERATIONS TO 20. 

# def run_purification_experiments(initial_state, T1, T2, max_iterations=20):
#     threshold_values = np.linspace(0.8, 0.99, 10)  # Fidelity threshold values
#     delay_times = np.linspace(0, 270e-4, 50)  # Delay times from 0 to 100e-4
    
#     failure_data = {}
    
#     for threshold in threshold_values:
#         failure_point = None
#         for delay in delay_times:
#             print(f"Testing threshold {threshold:.2f} with delay {delay:.5f}...")
#             purified_state, fidelity_history = dejmps(
#                 initial_state=initial_state,
#                 threshold_fidelity=threshold,
#                 T1=T1,
#                 T2=T2,
#                 delay_time=delay,
#                 max_iterations=max_iterations
#             )
            
#             if fidelity_history[-1] < threshold:
#                 failure_point = delay
#                 break
        
#         failure_data[threshold] = failure_point if failure_point is not None else max(delay_times)
    
#     return threshold_values, failure_data

# def plot_results(threshold_valuesDEJMPS, failure_dataDEJMPS, threshold_valuesBBPSSW, failure_dataBBPSSW ):
#     failure_pointsDEJMPS = [failure_dataDEJMPS[t] for t in threshold_valuesDEJMPS]
#     failure_pointsBBPSSW = [failure_dataBBPSSW[j] for j in threshold_valuesBBPSSW]

    
#     plt.figure(figsize=(8, 5))
#     plt.plot(threshold_valuesDEJMPS, failure_pointsDEJMPS, marker='o', linestyle='-', color='b', label = 'Deutsch Protocol')
#     plt.plot(threshold_valuesBBPSSW, failure_pointsBBPSSW, marker='o', linestyle='-', color='r', label='Bennett Protocol')
#     plt.xlabel("Fidelity Threshold")
#     plt.ylabel("Failure Delay Time (s)")
#     plt.title("Threshold Fidelity vs. Failure Network Latency Value")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('plots/bbpssw/bbpssw_fidelityIterationsVtime.svg', dpi=400)
#     plt.show()

# # Set initial state and memory parameters
# initial_state = bds_state(0.7)
# T1 = 1.14
# T2 = 0.5

# threshold_valuesDEJMPS, failure_dataDEJMPS = run_purification_experiments(initial_state, T1, T2)
# # print("Threshold values: ", threshold_values)
# # print("Failure Data: ", failure_data)
# threshold_valuesBBPSSW = [0.8,0.82111111, 0.84222222, 0.86333333, 0.88444444, 0.90555556, 0.92666667, 0.94777778, 0.96888889, 0.99]
# failure_dataBBPSSW = {np.float64(0.8): np.float64(0.01), np.float64(0.82111111): np.float64(0.01), np.float64(0.84222222): np.float64(0.009473684210526315), np.float64(0.86333333): np.float64(0.008421052631578947), np.float64(0.88444444): np.float64(0.007368421052631579), np.float64(0.90555556): np.float64(0.00631578947368421), np.float64(0.92666667): np.float64(0.004736842105263157), np.float64(0.94777778): np.float64(0.0036842105263157894), np.float64(0.96888889): np.float64(0.002105263157894737), np.float64(0.99): np.float64(0.0010526315789473684)}

# plot_results(threshold_valuesDEJMPS, failure_dataDEJMPS, threshold_valuesBBPSSW, failure_dataBBPSSW)


#--------------------------------------------------------------------------------------------------
# FIDELITY VS Purification Iterations

# def plot_fidelity_vs_time(all_fidelity_histories, threshold_fidelity, T1, T2):
#     plt.figure(figsize=(10, 6))
    
#     for delay_time, fidelity_history in all_fidelity_histories.items():
#         iterations = np.arange(0, len(fidelity_history))
#         plt.plot(iterations, fidelity_history, marker='o', linestyle='-', label=f'Delay {delay_time * 1e3:.3f} ms')
    
#     plt.axhline(y=threshold_fidelity, color='r', linestyle='--', label='Threshold Fidelity')
    
#     max_iterations = max(len(fh) for fh in all_fidelity_histories.values())
#     for i in range(1, max_iterations + 1):
#         plt.axvline(x=i, color='gray', linestyle='dotted', alpha=0.6)
    
#     plt.xlabel('Purification Iteration')
#     plt.ylabel('Fidelity')
#     plt.title(f'Deutsch Protocol: Fidelity vs Purification Iterations (T1={T1} s, T2={T2} s)')
#     plt.legend()
#     plt.grid(True, linestyle='dotted')
#     plt.savefig('plots/dejmps/dejmps_fidelityVsIterations200.svg', dpi=400)
#     plt.show()


# channel_lengths = np.linspace(10, 90, 9, endpoint=True)
# # channel_lengths = np.array([1e5, 1e4])
# speed_of_light = 2e5  # in fibre
# # delays = [0.0005, 0.001, 0.002, 0.003]
# delays = [0.001]
# bar_gr_result_f = {delay: [] for delay in delays}
# quantum_channel_lengths = [20, 22]
# # memory_params = {"T1": [86400, 1.14, 100, 3600, 600, 10000], "T2": [63, 0.5, 0.0018, 1.58, 1.2, 667]}
# memory_params = {"T1": [200], "T2": [0.5]}
# # memory_params = {"T1": [1.14], "T2": [0.5]}
# # memory_params = {"T1": [0.75], "T2": [0.18]}
# # memory_params = {"T1": [0.0256], "T2": [0.034]}

# # Run iterative purification and plot results
# threshold_fidelity_values = [0.95]
# delay_time_values = np.linspace(0, 130e-4, 13)
# initial_state = bds_state(0.7)
# for threshold_fidelity in threshold_fidelity_values:
#     all_fidelity_histories = {}
    
#     for delay_time in delay_time_values:
#         purified_state, fidelity_history = dejmps(
#             initial_state=initial_state,
#             threshold_fidelity=threshold_fidelity,
#             T1=memory_params["T1"][0],
#             T2=memory_params["T2"][0],
#             delay_time=delay_time,
#             max_iterations=10
#         )
        
#         if fidelity_history:
#             all_fidelity_histories[delay_time] = fidelity_history
    
#     if all_fidelity_histories:
#         plot_fidelity_vs_time(all_fidelity_histories, threshold_fidelity, memory_params["T1"][0], memory_params["T2"][0])