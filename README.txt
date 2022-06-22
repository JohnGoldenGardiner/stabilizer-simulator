AUTHOR: John Gardiner

REQUIREMENTS: Python 3.5 or higher

CODE STRUCTURE:  The file simulator.py defines five functions. The main one is
simulate(), which takes as input a qasm file, formatted as 

OPENQASM 2.0;
include "qelib1.inc";
qreg q[<number of qubits>];
creg c[<number of qubits>];
<first gate in the circuit>
<second gate in the circuit>
.
.
.
<last gate in the circuit>

where each gate is a line of the form

cx q[0],q[1];

for example. The simulator currently only allows for the single qubit 
gates h, x, t, tdg and the two-qubit gate cx, all notated thus. It 
outputs a 2**n by 1 numpy array whose entries represent the amplitudes 
of the quantum state that results from acting on |0...0> with the 
circuit. Here n is the number of qubits that are nontrivially involved
in the circuit. The output uses the same ordering convention for qubits
as the Cirq simulator and ignores any qubits that are unused in the 
circuit.

The other functions are auxiliary functions to simulate(). The function
qasm_string_to_gates() takes as input a qasm string and outputs a list
whose entries represent the gates in the circuit. The function
stabilizers_to_state() takes inputs representing the stabilizers and
destabilizers and coefficients in the stabilizer basis and outputs a 
numpy array whose entries represent the state described by the
coefficients and the stabilizer basis. See the comments within the code
for more detail. The function stabilizer_simulator() takes as input a 
list of gates (as output from qasm_string_to_gates() for example) and
outputs a numpy array whose entries represent the amplitudes of the 
quantum state obtained by acting on |0...0> with the circuit described
by the gates in the input list. There is one final function,
qasm_file_to_string(), which is not called by any other function, but is
included in the file because it can be useful for testing purposes.
This function takes as input a text file formatted according to the
grammar in a-subset-of-qasm.pdf on Bruinlearn and simply outputs the text
as a single string.

simulate() calls qasm_string_to_gates() and stabilizer_simulator().
stabilizer_simulator() calls stabilizers_to_state().
