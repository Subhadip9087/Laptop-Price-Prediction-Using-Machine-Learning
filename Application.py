import streamlit as st
import pickle
import numpy as np


# Import the Model
pipe = pickle.load(open('Pipe.pkl', 'rb'))
df = pickle.load(open('DataFrame.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of Laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS_Panel
ips_panel = st.selectbox('IPS Panel', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2800x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['CPU_Brand'].unique())

# Memory
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['GPU_Brand'].unique())

# Operating System
os = st.selectbox('Operating System', df['OS'].unique())

if st.button('Predict Price'):
    # Query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips_panel == 'Yes':
        ips_panel = 1
    else:
        ips_panel = 0

    X_Resolution = int(resolution.split('x')[0])
    Y_Resolution = int(resolution.split('x')[1])

    ppi = ((X_Resolution**2) + (Y_Resolution**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips_panel, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title("The Predicted Price of this Specified Laptop is " + str(int(np.exp(pipe.predict(query)[0]))))
