import pandas as pd
import streamlit as st


st.title("⚙️ AI-Powered Product Cost Predictor")
st.write("Leverage intelligent modeling to forecast costs for various production volumes. Just enter your product specs — we’ll do the math.")


with st.expander('Parameters'):
    Type = st.selectbox('Type:', ['Select type...','Door', 'Frame'])
    Color = st.selectbox('Color:', ['Select color...','RAL7024', 'RAL7035', 'RAL9005'])
    Handle = st.selectbox('Handle:', ['Select handle...','Aluminum (AL)', 'Steel (ST)', 'Plastic (PL)'])
    Hinge = st.selectbox('Hinge:', ['Select hinge...','Left', 'Right'])
    Packing = st.selectbox('Packing:', ['Select packing...','Wooden box', 'Cardbox'])
    Height = st.slider('Height (mm):', 1200, 1890, step=10)
    Width = st.slider('Width (mm):', 614, 750, step=10)