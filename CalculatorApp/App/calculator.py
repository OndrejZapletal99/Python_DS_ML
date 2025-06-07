import pandas as pd
import streamlit as st


st.title("⚙️ AI-Powered Product Cost Predictor")
st.write("Leverage intelligent modeling to forecast costs for various production volumes. Just enter your product specs — we’ll do the math.")


with st.expander('Parameters'):
    col1, col2, col3 = st.columns(3)

    with col1:
        Type = st.selectbox('Type:', ['Select type...', 'Door', 'Frame'])
        Handle = st.selectbox('Handle:', ['Select handle...', 'Aluminum (AL)', 'Steel (ST)', 'Plastic (PL)'])
        Height = st.slider('Height (mm):', 1200, 1890, step=10)

    with col2:
        Color = st.selectbox('Color:', ['Select color...', 'RAL7024', 'RAL7035', 'RAL9005'])
        Hinge = st.selectbox('Hinge:', ['Select hinge...', 'Left', 'Right'])
        Width = st.slider('Width (mm):', 614, 750, step=10)

    with col3:
        Packing = st.selectbox('Packing:', ['Select packing...', 'Wooden box', 'Cardbox'])

if st.button("Calculate Cost"):
    if 'Select packing...' in [Type, Color, Handle, Hinge, Packing]:
        st.warning("Please select all options before proceeding.")
    else:
        # Call your cost prediction logic here
        st.success("Parameters confirmed. Running prediction...")