import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Laptop Price Predection")
# Import model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

company=st.selectbox('Brand',df['Company'].unique())
typename=st.selectbox('typename',df['TypeName'].unique())
ram=st.selectbox('Ram',df['Ram'].sort_values(ascending=True).unique())
weight=st.number_input('Enter the weight')
ips=st.selectbox('IPS',['YES','NO'])
touchscreen=st.selectbox('Touchscreen',['YES','NO'])
fullhd=st.selectbox('Full HD',['YES','NO'])
screenresulution=st.selectbox('Screen Resolution',[
    '1920x1080',  
    '1366x768', 
    '1600x900',   
    '2560x1440',  
    '3840x2160', 
    '2880x1800', 
    '2560x1600',  
    '3200x1800',  
    '1440x900',   
    '1920x1200',  
    '2304x1440',  
    '5120x2880'  
])

screensize=st.number_input("Inches",min_value=10.0, max_value=20.0, value=15.6)

# calculate ppi
x_res=int(screenresulution.split('x')[0])
y_res=int(screenresulution.split('x')[1])
ppi = ((x_res**2 + y_res**2)**0.5) / screensize

ssd=st.selectbox('SSD in gb',[0, 8, 128, 256, 512, 1024, 2048])
hdd=st.selectbox('HDD in gb',[0, 128, 256, 512, 1024, 2048])

cup=st.selectbox('CPU',df['cpu brand'].unique())
os=st.selectbox('OS',df['OS'].unique())
gup=st.selectbox('GPU',df['new gpu'].unique())

if st.button('Predict Price'):
    ts = 1 if touchscreen == 'YES' else 0
    ips_val = 1 if ips == 'YES' else 0
    fhd = 1 if fullhd == 'YES' else 0

    query = [[company, typename, ram, weight,ips_val,fhd, ts , ppi,ssd,hdd,gup ,cup , os]]
    prediction=np.exp(pipe.predict(query))
    st.title(f"The predicted price of this laptop is: {int(prediction[0]*1.6)}")

    



