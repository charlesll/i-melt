import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# imelt
import imelt

# load the pre-trained 10 best models
neuralmodel = imelt.load_pretrained_bagged()

# streamlit preparation
st.title('i-Melt: prediction of melt and glass properties')
st.markdown('i-Melt is a multitask greybox model to predict the properties of glasses and melts.')
st.markdown('The published version allows predicting properties in the Na$_2$O-K$_2$O-Al$_2$O$_3$-SiO$_2$ diagram.')
st.markdown('The paper describing the model is available [here (open access)](https://www.sciencedirect.com/science/article/pii/S0016703721005007) and the code is hosted on [Github](https://github.com/charlesll/i-melt).')
# Session state callbacks
def callback_x1():
    """The callback renormalizes the values that were not updated."""
    # Here, if we modify X1, we calculate the remaining value for normalisation of X2, X3 and X4
    remain = 100 - st.session_state.x1
    # This is the proportions of X2, X3 and X4 in remain
    sum = st.session_state.x2 + st.session_state.x3 + st.session_state.x4
    # This is the normalisation step
    st.session_state.x2 = st.session_state.x2/sum*remain
    st.session_state.x3 = st.session_state.x3/sum*remain
    st.session_state.x4 = st.session_state.x4/sum*remain

def callback_x2():
    remain = 100 - st.session_state.x2
    sum = st.session_state.x1 + st.session_state.x3 + st.session_state.x4
    st.session_state.x1 = st.session_state.x1/sum*remain
    st.session_state.x3 = st.session_state.x3/sum*remain
    st.session_state.x4 = st.session_state.x4/sum*remain

def callback_x3():
    remain = 100 - st.session_state.x3
    sum = st.session_state.x1 + st.session_state.x2 + st.session_state.x4
    st.session_state.x1 = st.session_state.x1/sum*remain
    st.session_state.x2 = st.session_state.x2/sum*remain
    st.session_state.x4 = st.session_state.x4/sum*remain

def callback_x4():
    remain = 100 - st.session_state.x4
    sum = st.session_state.x1 + st.session_state.x2 + st.session_state.x3
    st.session_state.x1 = st.session_state.x1/sum*remain
    st.session_state.x2 = st.session_state.x2/sum*remain
    st.session_state.x3 = st.session_state.x3/sum*remain

# Sessions tate initialise
# Check if 'key' already exists in session_state
# If not, then initialize it
if 'x1' not in st.session_state:
    st.session_state['x1'] = 75.0

if 'x2' not in st.session_state:
    st.session_state['x2'] = 6.0

if 'x3' not in st.session_state:
    st.session_state['x3'] = 19.0

if 'x4' not in st.session_state:
    st.session_state['x4'] = 0.0

# The four sliders
st.subheader('Glass composition (move the sliders to change it!)')
st.slider("SiO2 concentration, mol%",
                min_value = 50.0,
                max_value = 100.0,
                key='x1', on_change=callback_x1)

st.slider("Al2O3 concentration, mol%",
                min_value = 0.0,
                max_value = 50.0,
                key='x2', on_change=callback_x2)

st.slider("Na2O concentration, mol%",
                min_value = 0.0,
                max_value = 50.0,
                key='x3', on_change=callback_x3)

st.slider("K2O concentration, mol%",
                min_value = 0.0,
                max_value = 50.0,
                key='x4', on_change=callback_x4)

composition =  np.array([st.session_state.x1/100, st.session_state.x2/100, st.session_state.x3/100, st.session_state.x4/100]).reshape(1,-1)

# PROPERTIES
tg = neuralmodel.predict("tg", composition)
density = neuralmodel.predict("density", composition)
ri = neuralmodel.predict("sellmeier", composition, [589.0])
sctg = neuralmodel.predict("sctg", composition)
fragility = neuralmodel.predict("fragility", composition)

# TVF parameters
A_VFT = neuralmodel.predict("a_tvf", composition)
B_VFT = neuralmodel.predict("b_tvf", composition)
C_VFT = neuralmodel.predict("c_tvf", composition)

# WORKING POINTS
WP_T = B_VFT/(3.-A_VFT) + C_VFT # working temperature
SP_T = B_VFT/(6.6-A_VFT) + C_VFT # softening temperature

# VISCOSITY CURVE
T_max = B_VFT/(0.-A_VFT) + C_VFT
T_min = B_VFT/(13.-A_VFT) + C_VFT
T_range = np.arange(T_min.mean(axis=1)[0], T_max.mean(axis=1)[0], 1.0)

raman = neuralmodel.predict("raman_pred", composition)

###
# Print useful parameters
###
st.subheader('Glass properties')
st.markdown('Glass transition temperature *Tg* = {:0.0f} K'.format(tg.mean()) + ' *(model 1$\sigma$ test error = 19 K)*')
st.markdown('Density = {:0.2f}'.format(density.mean())+' g cm$^{-3}$ *(model 1$\sigma$ test error 0.02 g cm$^{-3}$)*')
st.markdown('Optical refractive index at 589 nm = {:0.3f}'.format(ri.mean()) + ' *(model 1$\sigma$ test error 0.006)*')
st.markdown('Configurational entropy at *Tg* = {:0.1f}'.format(sctg.mean())+' J mol$^{-1}$ K$^{-1}$' + ' *(model 1$\sigma$ test error 0.9 J mol$^{-1}$ K$^{-1}$)*')
st.subheader('Melt properties')
st.markdown('Working point ($\eta$ = 3 log$_{10}$ Pa$\cdot$s) ='+' {:0.0f} K'.format(WP_T.mean()))
st.markdown('Softening point ($\eta$ = 6.7 log$_{10}$ Pa$\cdot$s) ='+' {:0.0f} K'.format(SP_T.mean()))
st.markdown('Fragility = {:0.1f}'.format(fragility.mean()))
st.markdown('$$log_{10} \eta$$ '+'= {:0.2f} + {:0.1f}/(T-{:0.1f}), with T in K'.format(A_VFT.mean(), B_VFT.mean(), C_VFT.mean()))

###
# FIGURE
###
st.subheader('Melt viscosity and glass Raman spectrum')
fig = make_subplots(rows=1, cols=2,subplot_titles=("Melt viscosity", "Glass Raman spectrum"))
labels_ = ["Adam-Gibbs", "Avramov-Milchev", "Free Volume", "VFT", "MYEGA"]
for count,i in enumerate(["ag", "am", "cg", "tvf", "myega"]):
    viscosity = neuralmodel.predict(i,composition*np.ones((len(T_range),4)),T_range.reshape(-1,1))
    fig.add_trace(
        go.Scatter(x=10000/T_range, y=viscosity.mean(axis=1),name=labels_[count], legendgroup=1),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(400.,1250.,1.0),y=raman.mean(axis=2).ravel(), name="Raman spectra", legendgroup=2),
    row=1, col=2
)

# Update xaxis properties
fig.update_xaxes(title_text=r'10<sup>4</sup>/T, K<sup>-1</sup>', row=1, col=1)
fig.update_yaxes(title_text=r"log<sub>10</sub> viscosity, Pa s", row=1, col=1)

# Update yaxis properties
fig.update_xaxes(title_text=r"Raman shift, cm<sup>-1</sup>", row=1, col=2)
fig.update_yaxes(title_text="Intensity, a.u.", row=1, col=2)
fig.update_layout(height=600, width=800)
st.plotly_chart(fig)
