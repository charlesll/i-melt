import streamlit as st
import pandas as pd
import numpy as np
#from plotly.subplots import make_subplots
import plotly.graph_objects as go

# imelt
import imelt

st.set_page_config(layout="wide")

# load the pre-trained 10 best models
neuralmodel = imelt.load_pretrained_bagged()

# streamlit preparation
st.title('i-Melt: Prediction of melt and glass properties')
st.markdown("""
            (c) Le Losq C. and co. 2021-2025
            
            i-Melt is a greybox model that uses physical equations and machine 
            learning to predict the properties of
            glasses and melts in the Na$_2$O-K$_2$O-MgO-CaO-Al$_2$O$_3$-SiO$_2$ system.
            
            For details
            [see the references](https://i-melt.readthedocs.io/en/latest/references.html)
            and [download the code](https://github.com/charlesll/i-melt)!

            Select a composition using the sliders on the sidebar (left) and see predictions below. 
            """)


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
    
if 'x5' not in st.session_state:
    st.session_state['x5'] = 0.0
    
if 'x6' not in st.session_state:
    st.session_state['x6'] = 0.0

def form_callback():
    sum = st.session_state.x1 + st.session_state.x2 + st.session_state.x3 + st.session_state.x4 + st.session_state.x5 + st.session_state.x6
    st.session_state.x1 = st.session_state.x1/sum*100.0
    st.session_state.x2 = st.session_state.x2/sum*100.0
    st.session_state.x3 = st.session_state.x3/sum*100.0
    st.session_state.x4 = st.session_state.x4/sum*100.0
    st.session_state.x5 = st.session_state.x5/sum*100.0
    st.session_state.x6 = st.session_state.x6/sum*100.0

with st.sidebar.form(key='my_form'):
    st.subheader('Glass composition')
    st.write("Enter your composition below (will be rescaled to ensure it sums to 100%.) " 
             "\nWarning : there is no safeguard, "
             "you can enter any value you want, and ask for model extrapolation. " 
             "Closely monitor predictive error bars to detect it."
             "Tip: Do not hit enter until you finish typing your composition.")
    st.number_input("SiO\u2082 concentration, mol%",
                    min_value = 0.0,
                    max_value = 100.0,
                    step = 0.5,
                    key='x1')

    st.number_input("Al\u2082O\u2083 concentration, mol%",
                    min_value = 0.0,
                    max_value = 100.0,
                    step = 0.5,
                    key='x2')

    st.number_input("Na\u2082O concentration, mol%",
                    min_value = 0.0,
                    max_value = 100.0,
                    step = 0.5,
                    key='x3')

    st.number_input("K\u2082O concentration, mol%",
                    min_value = 0.0,
                    max_value = 100.0,
                    step = 0.5,
                    key='x4')
                    
    st.number_input("MgO concentration, mol%",
                    min_value = 0.0,
                    max_value = 100.0,
                    key='x5')
                    
    st.number_input("CaO concentration, mol%",
                    min_value = 0.0,
                    max_value = 100.0,
                    step = 0.5,
                    key='x6')
    
    option = st.selectbox(
    'Viscosity equation?',
    ('Adam-Gibbs', 'TVF', 'MYEGA', 'Avramov-Milchev', 'Free Volume'))

    submit_button = st.form_submit_button(label='Calculate!', on_click=form_callback)
composition =  np.array([st.session_state.x1/100, st.session_state.x2/100, st.session_state.x3/100, st.session_state.x4/100, st.session_state.x5/100, st.session_state.x6/100]).reshape(1,-1)

composition = imelt.descriptors(pd.DataFrame(composition, columns=['sio2', 'al2o3', 'na2o', 'k2o', 'mgo', 'cao'])).values

# PROPERTIES
n_sample = 25
tg = neuralmodel.predict("tg", composition, sampling=True, n_sample=n_sample)
density = neuralmodel.predict("density_glass", composition, sampling=True, n_sample=n_sample)
ri = neuralmodel.predict("sellmeier", composition, lbd=np.array([589.0*1e-3,]), sampling=True, n_sample=n_sample)
sctg = neuralmodel.predict("sctg", composition, sampling=True, n_sample=n_sample)
fragility = neuralmodel.predict("fragility", composition, sampling=True, n_sample=n_sample)
liquidus = neuralmodel.predict("liquidus", composition, sampling=True, n_sample=n_sample)
elastic = neuralmodel.predict("elastic_modulus", composition, sampling=True, n_sample=n_sample)
cte = neuralmodel.predict("cte", composition, sampling=True, n_sample=n_sample)

aCpl = neuralmodel.predict("aCpl", composition, sampling=True, n_sample=n_sample)
ap = neuralmodel.predict("ap_calc", composition, sampling=True, n_sample=n_sample)
bCpl = neuralmodel.predict("bCpl", composition, sampling=True, n_sample=n_sample)

# TVF parameters
A_VFT = neuralmodel.predict("a_tvf", composition, sampling=True, n_sample=n_sample)
B_VFT = neuralmodel.predict("b_tvf", composition, sampling=True, n_sample=n_sample)
C_VFT = neuralmodel.predict("c_tvf", composition, sampling=True, n_sample=n_sample)

# parameters for other viscosity models
A_AG = neuralmodel.predict("ae", composition, sampling=True, n_sample=n_sample)
A_AM = neuralmodel.predict("a_am", composition, sampling=True, n_sample=n_sample)
A_CG = neuralmodel.predict("a_cg", composition, sampling=True, n_sample=n_sample)

B_AG = neuralmodel.predict("be", composition, sampling=True, n_sample=n_sample)
B_CG = neuralmodel.predict("b_cg", composition, sampling=True, n_sample=n_sample)

TO_CG = neuralmodel.predict("to_cg", composition, sampling=True, n_sample=n_sample)
C_CG = neuralmodel.predict("c_cg", composition, sampling=True, n_sample=n_sample)

# WORKING POINTS
WP_T = B_VFT/(3.-A_VFT) + C_VFT # working temperature
SP_T = B_VFT/(6.6-A_VFT) + C_VFT # softening temperature

# VISCOSITY CURVE
T_max = B_VFT/(0.-A_VFT) + C_VFT
T_min = B_VFT/(13.-A_VFT) + C_VFT
T_range = np.arange(T_min.mean(axis=1)[0], T_max.mean(axis=1)[0], 10.0)

###
# Print useful parameters
###

with st.expander("Notes on reported uncertainties (click to expand):"):
    st.markdown("""
                1-sigma uncertainties calculated using MC Dropout are provided either in parenthesis 
                for printed numbers or as shaded areas in figures. Due to speed and frugality considerations for this online calculator,
                here uncertainties are evaluated using a limited number of model forward pass (n=500). 
                
                For better uncertainties estimates, a larger number of forward pass may be necessary, and 
                scaling with conformal prediction may also be best (see article). A Jupyter notebook
                is provided on Github to highlight how to perform such operations. 

                Very high error bars may be observed if predictions are asked for compositions that are not in the training set.
                In this case, the model is extrapolating and uncertainties are expected to be large.
                """)
    

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Glass transition $T_g$',"{:0.0f}({:0.0f}) K".format(tg.mean(),tg.std()))
    st.metric('Softening point',"{:0.0f}({:0.0f}) K".format(SP_T.mean(),SP_T.std()))
    st.metric('Working point','{:0.0f}({:0.0f}) K'.format(WP_T.mean(),WP_T.std()))
    st.metric('Liquidus temperature','{:0.0f}({:0.0f}) K'.format(liquidus.mean(),liquidus.std()))
    
with col2:
    st.markdown('Liquid heat capacity, J/(mol K):')
    st.markdown('$${:0.1f}({:0.1f}) + {:0.4f}({:.4f}) T$$'.format(aCpl.mean(),aCpl.std(),bCpl.mean(),bCpl.std()))
    st.metric('Configurational entropy at $T_g$',"{:0.1f}({:0.1f}) J/(mol K)".format(sctg.mean(),sctg.std()))
    st.metric('Melt fragility','{:0.1f}({:0.1f})'.format(fragility.mean(),fragility.std()))
    
with col3:
    st.metric('Density',"{:0.2f}({:0.2f}) g/cm\u00b3".format(density.mean(),density.std()))
    st.metric('Elastic modulus',"{:.0f}({:.0f}) GPa".format(elastic.mean(),elastic.std()))
    st.metric('Refractive index at 589 nm',"{:.3f}({:.3f})".format(ri.mean(),ri.std()))
    st.metric('Thermal expansion',"{:.2f}({:.2f})".format(cte.mean(),cte.std())+r"x1e-6/K")
    
with st.expander("Viscosity Equations (click to expand):"):
    st.markdown("Vogel-Tamman-Fulcher")
    st.markdown("$$\\log_{{10}} \eta = {:0.2f} + \\frac{{{:0.0f}}}{{T-{:0.1f}}}$$".format(A_VFT.mean(),B_VFT.mean(),C_VFT.mean()))
    st.markdown("Adam-Gibbs")
    st.markdown("$$\\log_{{10}} \eta = {:0.2f} + \\frac{{{:0.0f}}}{{T*({:0.1f} + {:0.1f} \\ln(T/{:0.0f}) + {:0.4f}(T-{:0.0f}) )}}$$".format(A_AG.mean(),B_AG.mean(),sctg.mean(),ap.mean(),tg.mean(),bCpl.mean(),tg.mean()))
    st.markdown("MYEGA")
    st.markdown("$$\\log_{{10}} \eta = {:0.2f} + (12 - {:0.2f})\\frac{{{:0.0f}}}{{T}}e^{{([\\frac{{{:0.1f}}}{{(12 - {:0.2f})}}-1][\\frac{{{:0.0f}}}{{T}}-1])}}$$".format(A_AG.mean(),A_AG.mean(),tg.mean(),fragility.mean(),A_AG.mean(),tg.mean()))
    st.markdown("Avramov-Milchev")
    st.markdown("$$\\log_{{10}} \eta = {:0.2f} + (12 - {:0.2f})\\frac{{{:0.0f}}}{{T}}^{{(\\frac{{{:0.1f}}}{{(12 - {:0.2f})}})}}$$".format(A_AM.mean(),A_AM.mean(),tg.mean(),fragility.mean(),A_AM.mean()))
    st.markdown("Free Volume")
    st.markdown("$$\\log_{{10}} \eta = {:0.2f} + \\frac{{2 \\times{:0.1f}}}{{(T - {:0.1f} +\sqrt{{(T - {:0.1f})^2 + {:0.1f}*T}}}}$$".format(A_CG.mean(),B_CG.mean(),TO_CG.mean(),TO_CG.mean(),C_CG.mean()))
###
# FIGURES and Download
###

correspondance = {"Adam-Gibbs": "ag", "Avramov-Milchev": "am", "Free Volume": "cg", "VFT": "tvf", "MYEGA": "myega"}
labels_ = ["Temperature, K", correspondance[option]]

# Viscosity prediction with the choosen equation
viscosity = neuralmodel.predict(correspondance[option],
                                composition*np.ones((len(T_range),39)),
                                T_range.reshape(-1,1), sampling=True, n_sample=n_sample)
visco_mean = neuralmodel.predict(correspondance[option],
                                composition*np.ones((len(T_range),39)),
                                T_range.reshape(-1,1), sampling=False)
visco_lb = visco_mean - viscosity.std(axis=1)
visco_ub = visco_mean + viscosity.std(axis=1)

# record in dataframe for output
viscosity_dataframe = pd.DataFrame(np.vstack((T_range.ravel(),visco_mean.ravel())).T, 
                                            columns=["Temperature, K", option])

# Raman spectra predictions
raman_shift=np.arange(400.,1250.,1.0)
raman = neuralmodel.predict("raman_pred", composition, sampling=True, n_sample=50)

# calculate mean values and boundaries
raman_mean = neuralmodel.predict("raman_pred", composition, sampling=False).ravel()
raman_lb = raman_mean - raman.std(axis=2).ravel()
raman_ub = raman_mean + raman.std(axis=2).ravel()

# record in dataframe for output
raman_spectra = pd.DataFrame(np.vstack((raman_shift.ravel(), raman_mean)).T, 
                             columns=["Raman shift, cm-1","Intensity"])

#
# Make figures
#

# VISCOSITY
fig2 = go.Figure([
    go.Scatter(x=T_range, y=visco_mean, 
               line=dict(color='rgb(0,0,0)'), mode='lines',
               name=option, showlegend=True),
    go.Scatter(
        name='Upper Bound',
        x=T_range, y=visco_ub,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound',
        x=T_range, y=visco_lb,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
])

# set legend position
fig2.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

# Update axis properties
fig2.update_xaxes(title_text=r'Temperature, K')
fig2.update_yaxes(title_text=r"log<sub>10</sub> viscosity, Pa s")

fig3 = go.Figure([
    go.Scatter(x=raman_shift,y=raman_mean, name="Raman spectra", showlegend=False, 
               line=dict(color='rgb(0,0,0)'), mode='lines'),
    go.Scatter(
        name='Upper Bound',
        x= raman_shift, y=raman_ub,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound',
        x= raman_shift, y=raman_lb,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
])

# Update axis properties
fig3.update_xaxes(title_text=r"Raman shift, cm<sup>-1</sup>")
fig3.update_yaxes(title_text="Intensity, a.u.")

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig2, use_container_width=True)
    visco_csv = convert_df(viscosity_dataframe)
    st.download_button("Press to Download Viscosity Values",
        visco_csv,
        "viscosity.csv",
        "text/csv",
        key='visco-csv'
        )
with col6:
    st.plotly_chart(fig3, use_container_width=True)
    raman_csv = convert_df(raman_spectra)
    st.download_button("Press to Download Raman Values",
                        raman_csv,
                        "raman.csv",
                        "text/csv",
                        key='raman-csv')
