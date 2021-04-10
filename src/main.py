import streamlit as st
import numpy as np 
import pandas as pd 

#streamlit run [filename]

st.title('StackOverflow 2019 Developer Survey')
st.write('(This data only reflects 50 rows for performance purposes)')
st.subheader('Use the sidebar to filter the data set by student status. The below charts will auto-generate based on your selections.')