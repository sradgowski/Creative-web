import streamlit as st
import numpy as np 
import pandas as pd 
import docx2txt
import pdfplumber
import numpy as np

# importing Qiskit
from qiskit import IBMQ, BasicAer
# from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute

# import basic plot tools
from qiskit.visualization import plot_histogram

qubiuts = 7
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




#streamlit run [filename]
def write():
    text = ""
    menu = ["Home","Dataset","DocumentFiles","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
    if st.button("Process"):
        if docx_file is not None:
            file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
            st.write(file_details)
			# Check File Type
            if docx_file.type == "text/plain":
				# raw_text = docx_file.read() # read as bytes
				# st.write(raw_text)
				# st.text(raw_text) # fails
                st.text(str(docx_file.read(),"utf-8")) # empty
                raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
				# st.text(raw_text) # Works
                st.write(raw_text) # works
            elif docx_file.type == "application/pdf":
				# raw_text = read_pdf(docx_file)
                # st.write(raw_text)
                try:
                    with pdfplumber.open(docx_file) as pdf:
                        page = pdf.pages[0]
                        text = (page.extract_text())
                        st.write(text)
                except:
                        st.write("None")
					    
					
            elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				# Use the right file processor ( Docx,Docx2Text,etc)
                raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
                st.write(raw_text)
    print(text)
    circuit_to_run = wordToBV('Qiskit',qubiuts)#Secret Msg
    st.write(circuit_to_run[0].draw(output='mpl'))
    backend = BasicAer.get_backend('qasm_simulator')
    shots = 4096
    results = execute(circuit_to_run[::-1], backend=backend, shots=shots).result()
    answer = results.get_counts()

    st.write(plot_histogram(answer))
