from matplotlib import pyplot as plt
import numpy as np
import qutip as qt
from qutip import *
from qutip.measurement import measure_povm

def check_werner_r1(rho, tol=1e-6):
    """
    Check whether a 4x4 matrix rho corresponds to the Werner-type state:
        rho_W = r1 * |1><1| + (1-r1)/3 * (I4 - |1><1|)
    where |1> = (|01> - |10>)/sqrt(2).

    In the computational basis {|00>, |01>, |10>, |11>}, the matrix form is:

        rho_W = [[(1-r1)/3,        0,            0,          0],
                 [0,    (2*r1+1)/6, (1-4*r1)/6,   0],
                 [0,    (1-4*r1)/6, (2*r1+1)/6,   0],
                 [0,         0,          0,   (1-r1)/3]]

    Parameters
    ----------
    rho : 2D numpy array (shape (4,4))
        The density matrix to test.
    tol : float
        Numerical tolerance for consistency checks.

    Returns
    -------
    (is_werner, r1_value)
      is_werner : bool
          True if all derived r1 values match within 'tol'.
      r1_value  : float or None
          The average r1 if consistent, else None.
    """

    # Corner diagonals (should be (1-r1)/3):
    corner_val = rho[0, 0]  # same as rho[3,3] ideally
    # Center diagonals (should be (2*r1 + 1)/6):
    center_diag_val = rho[1, 1]  # same as rho[2,2] ideally
    # Off-diagonals in the center block (should be (1 - 4*r1)/6):
    center_offdiag_val = rho[1, 2]  # same as rho[2,1] ideally

    # Solve for r1 from each expression:
    # 1) corner_val = (1 - r1)/3  => r1 = 1 - 3 * corner_val
    r1_corner = 1.0 - 3.0 * corner_val

    # 2) center_diag_val = (2*r1 + 1)/6 => 2*r1 + 1 = 6*center_diag_val => r1 = 3*center_diag_val - 0.5
    r1_center_diag = 3.0 * center_diag_val - 0.5

    # 3) center_offdiag_val = (1 - 4*r1)/6 => 1 - 4*r1 = 6*center_offdiag_val => r1 = (1 - 6*center_offdiag_val)/4
    r1_center_offdiag = (1.0 - 6.0 * center_offdiag_val) / 4.0

    # Put them all in a list
    r1_candidates = [r1_corner, r1_center_diag, r1_center_offdiag]

    # Check if they are all close within 'tol'
    if (np.allclose(r1_corner, r1_center_diag, atol=tol) and
        np.allclose(r1_corner, r1_center_offdiag, atol=tol)):
        # If consistent, return average
        r1_mean = np.mean(r1_candidates)
        return True, r1_mean
    else:
        # Not consistent -> not a Werner state of this specific form
        return False, None


phi_plus = bell_state("00")
phi_minus = bell_state("01")
psi_plus = bell_state("10")
psi_minus = bell_state("11")

def Werner_state(F):
    """Returns Werner state of fidelity F.
    
    Keyword arguments:
    F -- fidelity of Werner state, range [0, 1]
    """
    if F < 0 or F > 1:
        raise Exception('Fidelity must be between 0 and 1.')
    
    state = F * psi_minus * psi_minus.dag() + (1 - F) / 3 * (phi_plus * phi_plus.dag() + phi_minus * phi_minus.dag() + psi_plus * psi_plus.dag())    
    return state

def apply_twirling(rho):
    """
    Apply twirling operation to convert a general two-qubit state into Werner state
    using the transformations from Bennett protocol.

    Parameters:
        rho (Qobj): Input two-qubit state

    Returns:
        Qobj: Werner state after twirling
    """
    # Define Pauli matrices and identity
    I2 = qt.qeye(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    # sz = qt.sigmaz()

    # Define K transformations from Bennett protocol
    u1 = (I2 + 1j * sx) / np.sqrt(2)
    u2 = (I2 - 1j * sy) / np.sqrt(2)
    u3 = (1j * qt.basis([2], 0) * qt.basis([2], 0).dag()) + qt.basis([2], 1) * qt.basis([2], 1).dag()
    u4 = I2

    K = []
    for u in [u1, u2, u3, u4]:
        K.append(qt.tensor(u, u))

    # Apply twirling operation
    bracket_term = qt.Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
    rho_w = qt.Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
    for i in range(4):
        term = K[i].dag() @ K[i].dag() @ rho @ K[i] @ K[i]
        bracket_term += term
    for j in range(3):
        rho_w += K[j].dag() @ bracket_term @ K[j]
    rho_w = rho_w / 12

    return rho_w



def calc_fidelity(state, p0 = qt.bell_state('11')):
    try:
        return (qt.fidelity(state, p0))
    except:
        return (qt.fidelity(state, qt.tensor(p0, p0)))



def generate_collapse_operators(T1, T2):
    gam = 1 / T1  # probability of a type 1 error

    dep = 1 / T2  # probability of a type 2 error

    # operators acting on A whilst B is in flight
    c_ops_partial = [np.sqrt(gam) * qt.tensor(qt.destroy(2), qt.qeye(2)),
                     np.sqrt(dep) * qt.tensor(qt.sigmaz(), qt.qeye(2))]

    # operators acting on both A and B once B has arrived
    c_ops_full = c_ops_partial + [np.sqrt(gam) * qt.tensor(qt.qeye(2), qt.destroy(2)),
                                  np.sqrt(dep) * qt.tensor(qt.qeye(2), qt.sigmaz())]

    return c_ops_partial, c_ops_full


def gen_entangled_state():
    state = qt.bell_state('11')
    return state


def apply_t1t2_noise_to_entangled_state(t1, t2):
    """
    Apply T1 and T2 noise to each qubit in a two-qubit entangled state with different times.

    :param qutip_state: QuTiP quantum object representing the two-qubit entangled state.
    :param t1: T1 relaxation time for both qubits.
    :param t2: T2 dephasing time for both qubits.
    :return: QuTiP quantum object representing the noisy state.
    """

    c_ops_partial, c_ops_full = generate_collapse_operators(t1, t2)

    # unitary hamiltonian dynamics is trivial
    H = qt.tensor(qt.qeye(2), qt.qeye(2))
    # delta = 0.1 * 2 * np.pi
    # H = delta * (qt.tensor(qt.sigmax(), qt.qeye(2)) + qt.tensor(qt.qeye(2), qt.sigmax()))
    # generate initial state
    p0 = gen_entangled_state()

    # construct options
    opts = qt.Options(atol=1e-10,
                      rtol=1e-8,
                      nsteps=1e5)

    output = []

    # evolve initial state whilst B is in flight (single step update)
    result = qt.mesolve(H, p0, np.linspace(0, 50e-3, 1000), c_ops_full, options=opts)
    pprime = result.states[-1]
    output.append(calc_fidelity(pprime, p0))
    twirled_states = []
    state_after_noise = []
    for state in result.states[1:None]:
        output.append(np.square(calc_fidelity(state, p0)))
        twirled_states.append(apply_twirling(state))
        state_after_noise.append(state)
    # print("1000th index: ", np.square(fidelity(twirled_states[-1], bell_state("11"))))
    # count=0
    # for state in twirled_states:
    #     print(count,"@ ",np.square(fidelity(state, bell_state("11"))))
    #     count= count+1
    # print("fidelity list complete")
    
    # a = np.linspace(0, 50e-3, 1000)
    # print(a[998])

    return output[998], twirled_states[998]


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
    H = qt.tensor(qt.qeye(2), qt.qeye(2))  # trivial Hamiltonian evolution
    t_list = np.linspace(0, waiting_time, 1000)
    result = qt.mesolve(H, state, t_list, c_ops_full)
    return result.states[-1]


def perform_bbpssw_purification(initial_state, threshold_fidelity, T1, T2, delay_time, max_iterations=10):
    """
    Perform iterative BBPSSW purification until fidelity exceeds threshold or max iterations are reached.
    
    Parameters:
        initial_state (Qobj): Initial two-qubit state to purify.
        threshold_fidelity (float): Fidelity threshold to stop purification.
        T1 (float): T1 relaxation time for LOCC noise.
        T2 (float): T2 dephasing time for LOCC noise.
        delay_time (float): Time delay for LOCC noise.
        max_iterations (int): Maximum number of purification iterations.
    
    Returns:
        Qobj: Final purified state.
        list: Fidelity history after each iteration.
    """
    current_state = initial_state.copy()
    print()
    fidelity_history = [0.7]
    
    for iteration in range(max_iterations):
        # Create two copies of the current state for purification
        rho_initial = qt.tensor(current_state, current_state)
        success = False
        # Attempt purification until successful
        while not success:
            print("------------------------------------------------------------------")        
            print(f"Iteration {iteration + 1}")
            print("is input werner: ", check_werner_r1(current_state))
            rho = rho_initial.copy()
            # print(rho)
            print("Fidelity before purification: ", (calc_fidelity(rho, bell_state("00"))))

            # Apply Y rotations
            U_Y0 = tensor(sigmay(), qeye(2), qeye(2), qeye(2))
            U_Y2 = tensor(qeye(2), qeye(2), sigmay(), qeye(2))
            rho = U_Y0 * rho * U_Y0.dag()
            rho = U_Y2 * rho * U_Y2.dag()
                   
            
            # Apply CNOT gates
            proj0 = basis(2, 0).proj()
            proj1 = basis(2, 1).proj()
            I = qeye(2)
            X = sigmax()
            # CNOT 0 -> 2
            U_CNOT02 = tensor(proj0, I, I, I) + tensor(proj1, I, X, I)
            rho = U_CNOT02 * rho * U_CNOT02.dag()
            # CNOT 1 -> 3
            U_CNOT13 = tensor(I, proj0, I, I) + tensor(I, proj1, I, X)
            rho = U_CNOT13 * rho * U_CNOT13.dag()
            
            # Measurement on qubits 2 and 3
            Z0 = ket2dm(basis(2, 0))
            Z1 = ket2dm(basis(2, 1))
            PZ = [
                tensor(I, I, Z0, Z0),
                tensor(I, I, Z1, Z1),
                tensor(I, I, Z0, Z1),
                tensor(I, I, Z1, Z0)
            ]
            outcome, rho_post = measure_povm(rho, PZ)
            
            if outcome in [0, 1]:
                success = True
                # Process the resulting state
                rho_final = rho_post.ptrace([0, 1])
                U_corr = tensor(sigmay(), qeye(2))
                purified_state = U_corr * rho_final * U_corr.dag()
                print("Fidelity post purification w/o LOCC noise",np.square(qt.fidelity(purified_state, bell_state('11'))))

                # Apply LOCC noise
                purified_state_after_wait = apply_locc_noise(purified_state, delay_time, T1, T2)
                fidelity = np.square(qt.fidelity(purified_state_after_wait, bell_state('11')))
                fidelity_history.append(fidelity)
                print(f"Fidelity after purification and LOCC noise: {fidelity}")
                # print(purified_state_after_wait)
                print("------------------------------------------------------------------")
                if fidelity > threshold_fidelity:
                    print('SUCCESSFUL')
                    return purified_state_after_wait, fidelity_history
                else:
                    current_state = apply_twirling(purified_state_after_wait)

            else:
                # print("------------------------------------------------------------------")
                print("Purification failed. Retrying...")
                # print("------------------------------------------------------------------")
                
    
    print(f"Max iterations ({max_iterations}) reached. Final fidelity: {fidelity_history[-1] if fidelity_history else 'N/A'}")
    return current_state, fidelity_history

#--------------------------------------------------------------------------------------------------
#NORMAL RUN

# channel_lengths = np.linspace(10, 90, 9, endpoint=True)
# # channel_lengths = np.array([1e5, 1e4])
# speed_of_light = 2e5  # in fibre
# # delays = [0.0005, 0.001, 0.002, 0.003]
# delays = [0.001]
# bar_gr_result_f = {delay: [] for delay in delays}
# quantum_channel_lengths = [20, 22]
# # memory_params = {"T1": [86400, 1.14, 100, 3600, 600, 10000], "T2": [63, 0.5, 0.0018, 1.58, 1.2, 667]}
# memory_params = {"T1": [1.14], "T2": [0.5]}
# # memory_params = {"T1": [0.0256], "T2": [0.034]}
# # memory_params = {"T1": [200], "T2": [0.5]}

# ''' refer to the following for T1 and T2 values:
#     https://www.aqt.eu/quantum-memory-lifetime/
#     https://www.nature.com/articles/nmat2420
#     https://arxiv.org/pdf/2005.01852
#     https://www.nature.com/articles/nphys4254
#     https://www.nature.com/articles/s41566-017-0007-1
#     https://www.science.org/doi/full/10.1126/science.1220513
#     https://www.nature.com/articles/s41566-017-0050-y#MOESM1
#     '''

# for i in range(len(memory_params['T1'])):
#     results = {delay: [] for delay in delays}
#     twirled_states = {delay: [] for delay in delays}
#     for delay in delays:
#         # twirled_states[delay] = apply_t1t2_noise_to_entangled_state(t1=memory_params["T1"][i], t2=memory_params["T2"][i])
#         twirled_states[delay] = Werner_state(0.7)

# # print("Starting Fidelity: ",np.square(calc_fidelity(twirled_states[delays[-1]])))

# # # print(list(map(check_werner_r1, twirled_states[delays[-1]])))

# delay = delays[-1]
# initial_state = twirled_states[delay]
# threshold_fidelity = 0.95  # Example threshold, adjust as needed
# purified_state, fidelity_history = perform_bbpssw_purification(
#     initial_state=initial_state,
#     threshold_fidelity=threshold_fidelity,
#     T1=memory_params["T1"][0],
#     T2=memory_params["T2"][0],
#     delay_time=100e-4,
#     max_iterations=50
# )
# # For a single purification attempt:
# perform_bbpssw_purification_direct()

# #--------------------------------------------------------------------------------------------------
# # CODE FOR CREATING MAX FIDELITY V LATENCY PLOT. SET MAX ITERATIONS TO 20. 

# def run_purification_experiments(initial_state, T1, T2, max_iterations=20):
#     threshold_values = np.linspace(0.8, 0.99, 10)  # Fidelity threshold values
#     delay_times = np.linspace(0, 100e-4, 20)  # Delay times from 0 to 100e-4
    
#     failure_data = {}
    
#     for threshold in threshold_values:
#         failure_point = None
#         for delay in delay_times:
#             print(f"Testing threshold {threshold:.2f} with delay {delay:.5f}...")
#             purified_state, fidelity_history = perform_bbpssw_purification(
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

# def plot_results(threshold_values, failure_data):
#     failure_points = [failure_data[t] for t in threshold_values]
    
#     plt.figure(figsize=(8, 5))
#     plt.plot(threshold_values, failure_points, marker='o', linestyle='-', color='b')
#     plt.xlabel("Fidelity Threshold")
#     plt.ylabel("Failure Delay Time (ms)")
#     plt.title("Bennett Protocol: Threshold Fidelity vs. Failure Delay Time")
#     plt.grid(True)
#     plt.savefig('plots/bbpssw/bbpssw_fidelityIterationsVtime.svg', dpi=400)
#     plt.show()

# # Set initial state and memory parameters
# initial_state = Werner_state(0.7)
# T1 = 1.14
# T2 = 0.5

# threshold_values, failure_data = run_purification_experiments(initial_state, T1, T2)
# print("Threshold values: ", threshold_values)
# print("Failure Data: ", failure_data)
# plot_results(threshold_values, failure_data)


# #--------------------------------------------------------------------------------------------------
# # FIDELITY VS Purification Iterations

def plot_fidelity_vs_time(all_fidelity_histories, threshold_fidelity, T1, T2):
    plt.figure(figsize=(10, 6))
    # print(all_fidelity_histories)
    for delay_time, fidelity_history in all_fidelity_histories.items():
        iterations = np.arange(0, len(fidelity_history))
        # print(fidelity_history[0])
        # print(iterations)
        plt.plot(iterations, fidelity_history, marker='o', linestyle='-', label=f'Delay {delay_time * 1e3:.3f} ms')
        
    
    plt.axhline(y=threshold_fidelity, color='r', linestyle='--', label='Threshold Fidelity')
    
    max_iterations = max(len(fh) for fh in all_fidelity_histories.values())
    for i in range(1, max_iterations + 1):
        plt.axvline(x=i, color='gray', linestyle='dotted', alpha=0.6)
    
    plt.xlabel('Purification Iteration')
    plt.ylabel('Fidelity')
    plt.title(f'Bennett Protocol: Fidelity vs Purification Iterations (T1={T1} s, T2={T2} s)')
    plt.legend()
    plt.grid(True, linestyle='dotted')
    plt.savefig('plots/bbpssw/bbpssw_fidelityVsIterations.svg', dpi=400)
    plt.show()


# memory_params = {"T1": [86400, 1.14, 100, 3600, 600, 10000], "T2": [63, 0.5, 0.0018, 1.58, 1.2, 667]}
memory_params = {"T1": [1.14], "T2": [0.5]}
# memory_params = {"T1": [0.0256], "T2": [0.034]}
# memory_params = {"T1": [200], "T2": [0.5]}

# Run iterative purification and plot results
threshold_fidelity_values = [0.95]
delay_time_values = np.linspace(0, 100e-4, 10)
for threshold_fidelity in threshold_fidelity_values:
    all_fidelity_histories = {}
    initial_state = Werner_state(0.7)
    for delay_time in delay_time_values:
        purified_state, fidelity_history = perform_bbpssw_purification(
            initial_state=initial_state,
            threshold_fidelity=threshold_fidelity,
            T1=memory_params["T1"][0],
            T2=memory_params["T2"][0],
            delay_time=delay_time,
            max_iterations=20 
        )
        
        if fidelity_history:
            all_fidelity_histories[delay_time] = fidelity_history

    if all_fidelity_histories:
        plot_fidelity_vs_time(all_fidelity_histories, threshold_fidelity, memory_params["T1"][0], memory_params["T2"][0])
    # print(all_fidelity_histories)
