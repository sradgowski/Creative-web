import streamlit as st
import numpy as np 
import pandas as pd 

#streamlit run [filename]
def write():
    st.title('IBM Quantum Creative Challenge')
    st.subheader('Using interference to create a steganography protocol')
    st.title('Idea')
    st.write('Generate Key using a QRNG, . We can use BB84 protocol for the communication channel. But the main challenge here is the stega encoder and cover data')
    st.write('Steganography is a technique hiding secret information within innocent-looking information')
    st.markdown(
        '''## Steganography
Quantum steganography is the art of secretly transmitting quantum information while disguising the fact that any secret communication is taking place. Like classical steganography, this involves embedding the hidden communication within a seemingly innocent “covertext,” which in the quantum case is a quantum state. [1].

## REFERENCES
[1] - https://www.sciencedirect.com/science/article/pii/B9780128194386000190''')



