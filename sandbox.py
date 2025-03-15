import numpy as np
import qutip as qt
from qutip import * 


channel_lengths = np.linspace(10, 90, 9, endpoint=True)
# channel_lengths = np.array([1e5, 1e4])
speed_of_light = 2e5  # in fibre
# delays = [0.0005, 0.001, 0.002, 0.003]
delays = [0.001]
bar_gr_result_f = {delay: [] for delay in delays}
quantum_channel_lengths = [20, 22]
# memory_params = {"T1": [86400, 1.14, 100, 3600, 600, 10000], "T2": [63, 0.5, 0.0018, 1.58, 1.2, 667]}
memory_params = {"T1": [0.0012], "T2": [0.00072]}

''' refer to the following for T1 and T2 values:
    https://www.aqt.eu/quantum-memory-lifetime/
    https://www.nature.com/articles/nmat2420
    https://arxiv.org/pdf/2005.01852
    https://www.nature.com/articles/nphys4254
    https://www.nature.com/articles/s41566-017-0007-1
    https://www.science.org/doi/full/10.1126/science.1220513
    https://www.nature.com/articles/s41566-017-0050-y#MOESM1
    '''

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
        return qt.fidelity(state, p0)
    except:
        return qt.fidelity(state, qt.tensor(p0, p0))


def generate_collapse_operators(T1, T2):
    gam = 1 - np.exp(- 1 / T1)  # probability of a type 1 error

    dep = 1 - np.exp(- 1 / T2)  # probability of a type 2 error

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


def apply_t1t2_noise_to_entangled_state(t1, t2, speed_of_light, delay):
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
    result = qt.mesolve(H, p0, np.linspace(0, 50e-3, 1000), c_ops_partial, options=opts)
    pprime = result.states[-1]
    output.append(calc_fidelity(pprime, p0))
    twirled_states = []
    state_after_noise = []
    for state in result.states[1:None]:
        output.append(np.square(calc_fidelity(state, p0)))
        twirled_states.append(apply_twirling(state))
        state_after_noise.append(state)

    return output, twirled_states[200]


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


mem_values = []
for i in range(len(memory_params['T1'])):
    results = {delay: [] for delay in delays}
    twirled_states = {delay: [] for delay in delays}
    for delay in delays:
        results[delay], twirled_states[delay] = apply_t1t2_noise_to_entangled_state(t1=memory_params["T1"][i], t2=memory_params["T2"][i], speed_of_light=speed_of_light, delay=delay)
    mem_values.append(results)

# print(list(map(check_werner_r1, twirled_states[delays[-1]])))

from qutip import *
import numpy as np
from qutip.measurement import measure_povm


def perform_bbpssw_purification_direct():
    # Get the list of Werner states (each on 4 qubits) from the final delay
    # rho_list = [tensor(rho, rho) for rho in twirled_states[delays[-1]]]
    rho_list = [tensor(twirled_states[delays[-1]],twirled_states[delays[-1]])]

    purified_states = []
    fidelity_list = []

    # Process each Werner state until purification succeeds.
    for original_rho in rho_list:
        success = False
        while not success:
            # Use a copy of the original state for this trial.
            rho = original_rho.copy()

            # --- Apply local Y rotations ---
            # Apply Y on qubit 0:
            U_Y0 = tensor(sigmay(), qeye(2), qeye(2), qeye(2))
            # Apply Y on qubit 2:
            U_Y2 = tensor(qeye(2), qeye(2), sigmay(), qeye(2))
            rho = U_Y0 * rho * U_Y0.dag()
            rho = U_Y2 * rho * U_Y2.dag()

            # --- Apply CNOT gates ---
            # Define projectors and operators.
            proj0 = basis(2, 0) * basis(2, 0).dag()
            proj1 = basis(2, 1) * basis(2, 1).dag()
            I = qeye(2)
            X = sigmax()

            # CNOT with control = qubit 0, target = qubit 2:
            U_CNOT_02 = tensor(proj0, I, I, I) + tensor(proj1, I, X, I)
            rho = U_CNOT_02 * rho * U_CNOT_02.dag()
            # CNOT with control = qubit 1, target = qubit 3:
            U_CNOT_13 = tensor(I, proj0, I, I) + tensor(I, proj1, I, X)
            rho = U_CNOT_13 * rho * U_CNOT_13.dag()

            # --- Measurement on qubits 2 and 3 ---
            # Define measurement operators for the Z basis on qubits 2 and 3.
            Z0 = ket2dm(basis(2, 0))
            Z1 = ket2dm(basis(2, 1))
            PZ = [
                tensor(qeye(2), qeye(2), Z0, Z0),
                tensor(qeye(2), qeye(2), Z1, Z1),
                tensor(qeye(2), qeye(2), Z0, Z1),
                tensor(qeye(2), qeye(2), Z1, Z0)
            ]
            outcome, rho_post = measure_povm(rho, PZ)

            if outcome in [0, 1]:
                # Successful outcome.
                # Trace out the measured qubits (2 and 3) to obtain a 2-qubit state.
                rho_final = rho_post.ptrace([0, 1])
                # Apply the corrective unitary: Y on qubit 0.
                U_corr = tensor(sigmay(), qeye(2))
                purified_state = U_corr * rho_final * U_corr.dag()
                # Now simulate waiting noise during the classical communication delay.
                purified_state_after_wait = apply_locc_noise(purified_state, 50e-3, 0.0012, 0.00072)

                purified_states.append(purified_state_after_wait)

                # Compute and store the fidelity with respect to the target Bell state (psi_minus).
                fidelity_list.append(np.square(fidelity(purified_state_after_wait, bell_state("11"))))
                print("Final Fidelity for this state:", fidelity_list[-1])
                success = True  # Exit the while-loop for this state.
            else:
                print("Purification failed for this trial (measurement outcome not 0 or 1). Retrying...")

    return purified_states, fidelity_list


# For a single purification attempt:
perform_bbpssw_purification_direct()