# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 18:52:26 2022

@author: JohnGardiner
"""



import numpy as np



"""
Turns a qasm file, in the proper format, into a "qasm string". Useful for
testing purposess. Enter name of file as a string
"""
def qasm_file_to_string(qasm_file):
    qasm_string = open(qasm_file, 'r').read()
    return qasm_string



"""
Turns a "qasm string" into a list of gates.
"""
def qasm_string_to_gates(qasm_string):
    
    #Split into lines. Then split each line into gate name then qubit name.
    lines_raw = qasm_string.split(';\n')
    if lines_raw[-1] == '':
        lines_raw.pop()
    lines = [i.replace(',', ' ').replace('q[', ' ').replace(']', ' ').replace('c[', ' ') for i in lines_raw]
    lines = [i.split() for i in lines]
    
    #Check that input file is formatted the way the assignment requires
    assert lines[0] == ['OPENQASM', '2.0'], 'Input file incorrectly formatted'
    assert lines[1] == ['include', '"qelib1.inc"'], 'Input file incorrectly formatted'
    assert lines[2][0] == 'qreg', 'Input file incorrectly formatted'
    assert lines[3][0] == 'creg', 'Input file incorrectly formatted'
    
    #Make the list of gates
    gate_list = lines[4:]
    
    return gate_list



"""
Takes as input destabilizer and stabilizer group generators, which specify a
basis, as well as the coefficients of a quantum state in that basis, and
outputs that quantum state in the computational basis. D_x, D_z, and D_sign
describe the destabilizers; S_x, S_z, and S_sign describe the stabilizers; chi
are the coefficients; nonzero_chi is a list of indices of nonzero elements of
chi (info convenient for making the program run faster); and seed_state is a
quantum state that will be acted on with the projector|psi><psi|, where |psi>
is the stabilizer state. It is not so important what seed_state is, only that
its overlap with the stabilizer state |psi> not be zero. If we make an unlucky
choice of seed_state, such that acting on it with |psi><psi| gives zero, the
procedure will start over with a new randomly chosen seed_state. The output of
this function is a numpy array representing a normalized quantum state.
"""
def stabilizers_to_state(D_x, D_z, D_sign, S_x, S_z, S_sign, chi, nonzero_chi, seed_state):

    #Initialize some values
    n = len(S_x)
    stabilizer_state = seed_state
    
    """
    The inputs S_x, S_z are lists where each entry (read in binary) represents
    a row of a matrix whose columns describe the n stabilizers. For easier
    computation we define new lists s_x, s_z that describe the same stabilizers
    in a different way. Namely the entries of s_x and s_z (read in binary)
    describe the columns of the aforementioned matrix, so each entry
    corresponds to a stabilizer. So for example, s_x[i] and s_z[i] together
    describe the i-th generator of the stabilizer group.
    """
    s_x = [sum(((S_x[j]>>i)%2)<<j for j in range(n)) for i in range(n)]
    s_z = [sum(((S_z[j]>>i)%2)<<j for j in range(n)) for i in range(n)]
    
    """
    Project |seed_state> onto the stabilizer state. The projector, up to
    normalization, is (1+g_n)...(1+g_2)(1+g_1) where g_i are the n stabilizer
    group generators. This loop repeatedly updates |stabilizer_state> with
    (1+g_i)|stabilizer_state>
    """
    integer_type = 'int64'
    if n < 7:
        integer_type = 'int32'
    if n >= 15:
        integer_type = 'float'
    for i in range(n):
        stabilizer_state = stabilizer_state + np.array(
            [
                (-1)**(
                    sum((s_z[i]&(k^s_x[i]))>>j for j in range(n))
                    + (S_sign>>i)%2
                    )
                *stabilizer_state[k^s_x[i]]
                for k in range(2**n)
                ],
            dtype = integer_type)

    """
    Here we check that we haven't projected to zero. If we have (by an unlucky
    choice for seed_state) we start again with a new, randomly chosen
    seed_state
    """
    normsquared = sum(stabilizer_state**2)
    if normsquared == 0:
        #We start over with random seed_state
        seed_state = list(np.random.randint(-1,2,size=2**n))
        state = stabilizers_to_state(D_x, D_z, D_sign, S_x, S_z, S_sign, chi, nonzero_chi, seed_state)
        return state
    
    """
    Here we construct the stabilizer basis states and add take their linear
    combination with coefficients chi. This results in the desired state. The
    loop is over the stabilizer basis states that have nonzero coefficients. We
    add chi[a] * |basis state a> each time. The basis states are defined as
    h_1**a_1 h_2**a_2 ... h_n**a_n |stabilizer state>, where h_i are the
    destabilizer group generators and a_i are the bits of a.
    """
    state = 0
    for a in nonzero_chi:

        #Multiply h_1**a_1 ... h_n**a_n and describe result by tot_x, tot_z, tot_sign
        #Find the number of Xs on each qubit
        tot_x = sum((sum(((D_x[i]&a)>>j)%2 for j in range(n))%2)<<i for i in range(n))
        #Find the number of Zs on each qubit
        tot_z = sum((sum(((D_z[i]&a)>>j)%2 for j in range(n))%2)<<i for i in range(n))
        #Find the resulting phase after anticommutation
        xorsum = D_sign&a
        for i in range(n):
            for j in range(1,n):
                xorsum = xorsum^(D_x[i]&a&((D_z[i]&a)<<j))
        tot_sign = sum((xorsum>>i)%2 for i in range(n))%2
        
        #Add chi[a] * |basis state a>
        state = state + chi[a]*np.array(
            [
                (-1)**(sum((tot_z&(k^tot_x))>>j for j in range(n)) + tot_sign)
                *stabilizer_state[k^tot_x]
                for k in range(2**n)
                ]
            )

    #Normalize the state
    norm = np.linalg.norm(state)
    state = state/norm
    
    return state



"""
This function takes as input a list of gates and outputs a list of amplitudes.
Each "gate" is itself a list, of the format [<string 'h', 'x', 'cx', etc.>,
<integer>, ...], where the first entry names the gate type and the remaining
entries specify which qubits the gate acts on. This is the heart of the
simulator and is called by the function simulate().
"""
def stabilizer_simulator(gate_list):
    
    #Determine which qubits are ever acted on in the circuit
    vertices = []
    for gate in gate_list:
        vertices.append(gate[1])
        if gate[0] == 'cx':
            vertices.append(gate[2])
    vertices = set(vertices)
    vertices = sorted([int(i) for i in vertices]) #Order qubit names numerically
    vertices = [str(i) for i in vertices]
    #Reindex the qubits to skip qubits that are unused in the circuit
    vertex_map = {j:i for i, j in enumerate(vertices)}
    
    #The number of qubits used in the circuit
    n = len(vertices)
    
    """
    Initialize the bit matrices representing the list of stabilizer and
    "destabilizer" group generators. Each element in the lists D_x, D_z, S_x,
    and S_z and S should be understood bitwise as a row of a matrix. These four
    matrices can be viewed as blocks in a single matrix like so:
    [       |       ]
    [  D_x  |  S_x  ]
    [_______|_______]
    [       |       ]
    [  D_z  |  S_z  ]
    [       |       ]
    The columns of said matrix represent the Paulis in the (de)stabilizer
    group generators. The rows of said matrix represent the different qubits.
    An entry of 1 in the ith row jth column of the D_x matrix signifies that
    a pauli X acting on the jth qubit is present in the ith generator of the
    destabilizer group. An entry of 1 in the ith row jth column of the S_z
    matrix signifies that a pauli Z acting on the jth qubit is present in the
    ith generator of the stabilizer group. Etc. D_sign and S_sign are integers
    whose bits keep track of the sign out front of the Paulis that make up the
    (de)stabilizer group generators. 0 represents + and 1 represents -.
    """
    #Initialize list of "destabilizer" group elements
    D_x = [2**i for i in range(n)]
    D_z = [0 for i in range(n)]
    D_sign = 0
    #Initialize list of stabilizer group elements.
    S_x = [0 for i in range(n)]
    S_z = [2**i for i in range(n)]
    S_sign = 0

    """
    The destabilizer and stabilizer group generators together specify a basis
    for the space of quantum states. We describe the quantum state by listing
    the coefficients relative to this basis. These coefficients are stored in
    the vector chi.
    """
    #Initialize a list of coefficients in front of stabilizer basis states.
    chi = np.array([0.0+0.0j for i in range(2**n)])
    chi[0] = 1
    #nonzero_chi keeps track of which coefficients are zero, to save time.
    nonzero_chi = {0}
    
    
    """
    Implement the gates. There are two ways to describe the action of a gate on
    the state. Either we can change the coefficients while leaving the basis
    states unchanged, or we can change the basis states while leaving the
    coefficients unchanged. H, X, and CX gates have a simple action on the
    basis states (in terms of our description of them via (de)stabilizers), so
    we implement them by updating the (de)stabilizer group generators. T gates
    on the other hand do not act in a way that is easily described by updating
    the (de)stabilizer group generators, so we implement T gates by updating
    the coefficients chi. Similarly for inverse T gates
    """
    for current_gate in gate_list:
        
        #Make sure the qasm file only includes 'h', 'x', 'cx', 't', 'tdg' gates
        assert isinstance(current_gate[0], str), 'Incorrectly formatted gate encountered in qasm string.'
        assert current_gate[0] in ['h', 'x', 'cx', 't', 'tdg'], 'The gate ' + current_gate[0] + ' is not in the accepted gate list.'
        
        #H gates
        if current_gate[0] == 'h':
            k = vertex_map[current_gate[1]]
            #Switch X operators to Z operators and vice versa
            D_x[k], D_z[k] = D_z[k], D_x[k]
            S_x[k], S_z[k] = S_z[k], S_x[k]
            #Change sign to account for possible anti-commutation
            D_sign = D_sign^(D_x[k]&D_z[k])
            S_sign = S_sign^(S_x[k]&S_z[k])

        #X gates
        if current_gate[0] == 'x':
            k = vertex_map[current_gate[1]]
            #Flip the sign whenever there is a Z in row k.
            D_sign = D_sign^D_z[k]
            S_sign = S_sign^S_z[k]

        #CX gates
        if current_gate[0] == 'cx':
            control = vertex_map[current_gate[1]]
            target = vertex_map[current_gate[2]]
            #Propagate X operators from control to target and propagate Z operators from target to control
            D_x[target] = D_x[target]^D_x[control]
            D_z[control] = D_z[control]^D_z[target]
            S_x[target] = S_x[target]^S_x[control]
            S_z[control] = S_z[control]^S_z[target]


        #T gates and inverse T gates
        if current_gate[0] == 't' or current_gate[0] == 'tdg':

            k = vertex_map[current_gate[1]]
            
            """
            For T and inverse T gates, we don't update the (de)stabilizer group
            generators. Instead, we update the coefficients that describe the
            state in the stabilizer basis.
            """
            
            """
            Note that a T gate can be decomposed as a sum of Paulis as
            T = cos(pi/8) I - i sin(pi/8) Z.
            So the action of a T gate on the coefficients is determined by how
            a Z gate acts on the coefficients. The first step to finding how a
            Z acts on the coefficients is to find the expansion of I...IZI...I
            in terms of destabilizer and stabilizer group generators. The
            values a and b and alpha describe this expansion. alpha describes
            the sign, the bits of a describe which destabilizer generators are
            present, and the bits of b describe which stabilizer generators are
            present. We can deduce a and b by noting that I...IZI...I will
            anticommute with a stabilizer generator for every destabilizer
            generator that is present in its expansion, and vice versa.
            """
            #Find a and b
            a = S_x[k]
            b = D_x[k]
            
            #Find alpha
            DaSb_x = [(D_x[i]&a) + ((S_x[i]&b)<<n) for i in range(n)]
            DaSb_z = [(D_z[i]&a) + ((S_z[i]&b)<<n) for i in range(n)]
            DaSb_sign = (D_sign&a) + ((S_sign&b)<<n)
            
            xorsum = DaSb_sign
            for j in range(1,2*n):
                for i in range(n):
                    xorsum = xorsum^(DaSb_x[i]&(DaSb_z[i]<<j))
            
            alpha = sum((xorsum>>i)%2 for i in range(2*n))%2
            
            #Update coefficients in chi
            eps = 1e-10 #We round small values to zero. This is the tolerance
            c1 = np.cos(np.pi/8) #We'll expand gate as c1 I + c2 Z
            if current_gate[0] == 't':
                c2 = -1j*np.sin(np.pi/8)
            else:
                c2 = 1j*np.sin(np.pi/8)
            newly_nonzero_chi = set()
            newly_zero_chi = set()
            for i in nonzero_chi:
                phase = (-1)**(alpha + sum(((b&i)>>j)%2 for j in range(n)))
                #The logic here is necessary to avoid updating values twice
                if i < i^a or (i^a not in nonzero_chi):
                    chi[i], chi[i^a] = c1*chi[i] + c2*phase*chi[i^a], c1*chi[i^a] + c2*phase*chi[i]
                    #We round small values to zero then remove them from the set nonzero_chi if they are zero. 
                    if np.isclose(chi[i], 0, atol=eps):
                        chi[i] = 0
                    if np.isclose(chi[i^a], 0, atol=eps):
                        chi[i^a] = 0
                    #Remove coefficients from the nonzero list if they are zero
                    if chi[i^a] == 0:
                        newly_zero_chi.add(i^a)
                    if chi[i] == 0:
                        newly_zero_chi.add(i)
                    #Add coefficients to the nonzero list if they are nonzero
                    if chi[i^a] != 0:
                        newly_nonzero_chi.add(i^a)
                if a == 0:
                    chi[i] = c1*chi[i] + c2*phase*chi[i]
            #Update nonzero chi
            nonzero_chi = nonzero_chi.difference(newly_zero_chi)
            nonzero_chi = nonzero_chi.union(newly_nonzero_chi)
    
        
    """
    Construct state. Here we call the function stabilizers_to_state()
    elsewhere defined, which takes as input the (de)stabilizer group generators
    and the coefficients and outputs the corresponding quantum state.
    """
    #Some slight time saving by using integers if we can get away with it
    integer_type = 'int64'
    if n < 7:
        integer_type = 'int32'
    if n >= 15:
        integer_type = 'float'
    #seed_state is just some state or other, required as input
    seed_state = np.ones((2**n), dtype=integer_type)
    state_vector = stabilizers_to_state(D_x, D_z, D_sign, S_x, S_z, S_sign, chi, nonzero_chi, seed_state)
    
    
    """
    Throughout we have used the convention that smaller powers of two represent
    earlier qubits. Cirq uses the opposite convention, so we make the switch
    here.
    """
    state_vector = np.array([
        state_vector[sum(((i>>j)%2)<<(n-1-j) for j in range(n))]
        for i in range(2**n)
        ])

    return state_vector



"""
This function takes as input a "qasm string" and outputs the resulting quantum
state. It simply changes the qasm string to a list of gates and feed that as
input to the stabilizer_simulator funtion defined above.
"""
def simulate(qasm_string):
    gate_list = qasm_string_to_gates(qasm_string)
    state_vector = stabilizer_simulator(gate_list)
    return state_vector
    
    
    
    
    
    
    