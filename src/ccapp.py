import streamlit as st
import numpy as np 
import pandas as pd 
import docx2txt
import pdfplumber
import numpy as np
from qiskit import *
# importing Qiskit
from qiskit import IBMQ, BasicAer
# from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute


# import basic plot tools
from qiskit.visualization import plot_histogram

def wordToBV(s, n) :
    #convert text to binary
    a_byte_array = bytearray(s, "utf8")
    byte_list = []


    for byte in a_byte_array:
        binary_representation = bin(byte)
        byte_list.append(binary_representation[9-n:])
        #chop off the "0b" at the beginning. can also truncate the binary to fit on a device with N qubits
        #binary has 2 extra digits for "0b", so it starts at 9 for our 7 bit operation. 

    print(byte_list)
    
    circuit_array = []
    
    length = len(byte_list) 
    
    for i in range(length):
    
        s = byte_list[i]


        #do all  this stuff for every letter

        # We need a circuit with n qubits, plus one ancilla qubit
        # Also need n classical bits to write the output to
        bv_circuit = QuantumCircuit(n+1, n)

        # put ancilla in state |->
        bv_circuit.h(n)
        bv_circuit.z(n)

        # Apply Hadamard gates before querying the oracle
        for i in range(n):
            bv_circuit.h(i)

        # Apply barrier 
        bv_circuit.barrier()

        # Apply the inner-product oracle
        s = s[::-1] # reverse s to fit qiskit's qubit ordering
        for q in range(n):
            if s[q] == '0':
                bv_circuit.i(q)
            else:
                bv_circuit.cx(q, n)

        # Apply barrier 
        bv_circuit.barrier()

        #Apply Hadamard gates after querying the oracle
        for i in range(n):
            bv_circuit.h(i)

        # Measurement
        for i in range(n):
            bv_circuit.measure(i, i)
            
        circuit_array.append(bv_circuit)

    
    return circuit_array

def encrypt(BB84_key, letter):
    """Calculates XOR"""
    if BB84_key is "":
        BB84_key="0000"
    b = int(BB84_key, 2)
    x = ord(letter)
    return format(b ^ x, "b")


def stega_encoder(LM, carrier_msg):
    """Encodes LM bits message into carrier_msg"""
    message = ""
    size = len(LM[0])
    i = 0
    for j, bitstring in enumerate(LM):
        for k, digit in enumerate(bitstring):
            while (not carrier_msg[i].isalpha()):
                message += carrier_msg[i]
                i += 1

            if digit == "1":
                letter = carrier_msg[i].upper()
                message += letter
            else:
                message += carrier_msg[i]

            i += 1
    
    if i < len(carrier_msg):
        message += carrier_msg[i:]

    return message


def stega_decoder(new_carrier_msg, BB84_key):
    """Decodes secret message from new_carrier_msg"""

    b = int(BB84_key, 2)

    message = ""
    bitstring = ""
    for char in new_carrier_msg:
        if char.isalpha():
            if char.isupper():
                bitstring += "1"
            else:
                bitstring += "0"

        if len(bitstring) == 7:
            x = int(bitstring, 2)
            message += chr(b ^ x)
            bitstring = ""

    return message
def sifted_key(A_basis,B_basis,key): 
 correct_basis=[]
 sifted_key=''

 for i in range(len(A_basis)):
  if A_basis[i]==B_basis[i]:
    correct_basis.append(i)
    sifted_key+=key[i]
  else:
    pass 
 return sifted_key,correct_basis


def write():
    text = ""
    menu = ["Home","Dataset","DocumentFiles","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    st.header('Set the number of Qubits ')
    qubits = st.slider('please choose 7 for the time being', min_value=1, max_value=10)
    qr = QuantumRegister(qubits, name='qr')
    cr = ClassicalRegister(qubits, name='cr')    
    qc = QuantumCircuit(qr, cr, name='QC')

    # BB84 Protocol :
    # Generate a random number in the range of available qubits [0,65536))
    alice_key = np.random.randint(0,2**qubits)#here we can remplace by a key from a quantum key generator


    alice_key = np.binary_repr(alice_key,qubits) 
    for i in range(len(alice_key)):
        if alice_key[i]=='1':
            qc.x(qr[i])
    B=[]
    for i in range(len(alice_key)):

        if 0.5 < np.random.random():
            qc.h(qr[i])
            B.append("H")
        else:
            B.append("S")
            pass
    
    qc.barrier()  
    print("Alice Basis",B)
    C=[]
    for i in range(len(alice_key)):
        if 0.5 < np.random.random():
            qc.h(qr[i])
            C.append("H")
            
        else:
            C.append("S")
        qc.barrier()
        for i in range(len(alice_key)):
            qc.measure(qr[i],cr[i])
            print("Bob Basis",C)
    simulator = Aer.get_backend('qasm_simulator')
    execute(qc, backend = simulator)
    result = execute(qc, backend = simulator).result()
    print("Bob key :",list(result.get_counts(qc))[0])
    print("Bob Basis",C)

    print("Alice key :",alice_key)
    print("Alice Basis :",B)
    a=sifted_key(B,C,alice_key)
    BB84_key=a[0]

    st.header('Secret Message')
    message = st.text_input('let the message be qiskit')
    if message:
        circuit_to_run = wordToBV(message,qubits)#Secret Msg
        st.write(circuit_to_run[0].draw(output='mpl'))
        backend = BasicAer.get_backend('qasm_simulator')
        shots = 4096
        results = execute(circuit_to_run[::-1], backend=backend, shots=shots).result()
        answer = results.get_counts()
        st.write(answer)
        st.write(plot_histogram(answer))
    encrypt(BB84_key,'q')
    st.header('Upload a sentence where you wish to hide your message ')
    st.write('')
    text_ob = st.text_input('let the sentence be hellooo worlddd  hellooo worlddd hellooo worlddd')
    if text_ob:
        L=[]
        for c in message:
            L.append(encrypt(BB84_key,c))
            new_carrier_msg=stega_encoder(L, text_ob)
        new_carrier_msg=stega_encoder(L, text_ob )
        st.header('Encoded message')
        if st.button('Encode'):
            st.write(new_carrier_msg)
        st.header('Decoded Message')
        if st.button('Decode'):
            st.write(stega_decoder(new_carrier_msg, BB84_key))
