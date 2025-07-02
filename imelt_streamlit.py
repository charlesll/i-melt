import streamlit as st
import pandas as pd
import numpy as np
#from plotly.subplots import make_subplots
import plotly.graph_objects as go

# imelt
import imelt

st.set_page_config(page_title="i-Melt",layout="wide")

# load the pre-trained 10 best models
@st.cache_resource  # Add the caching decorator
def load_model():
    return imelt.load_pretrained_bagged()
neuralmodel = load_model()

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

# Add information about the app
st.sidebar.info("Enter your composition below (it will be rescaled to ensure it sums to 100%). Warning : there is no safeguard, you can enter any value you want, and ask for model extrapolation. Closely monitor predictive error bars to detect it. Click on the 'Calculate Properties' button to see the results.")

st.sidebar.markdown('---')

# Sidebar for composition input type selection
composition_type = st.sidebar.radio("Select composition input type:", ("wt%", "mol%"))

# Sidebar for inputs
st.sidebar.header(f'Composition Input ({composition_type})')
composition = {
    'SiO2': st.sidebar.number_input('SiO2', value=60.0, min_value=0.0, max_value=100.0),
    'Al2O3': st.sidebar.number_input('Al2O3', value=10.0, min_value=0.0, max_value=100.0),
    'Na2O': st.sidebar.number_input('Na2O', value=5.0, min_value=0.0, max_value=100.0),
    'K2O': st.sidebar.number_input('K2O', value=5.0, min_value=0.0, max_value=100.0),
    'MgO': st.sidebar.number_input('MgO', value=10.0, min_value=0.0, max_value=100.0),
    'CaO': st.sidebar.number_input('CaO', value=10.0, min_value=0.0, max_value=100.0),
}

equation_ = st.sidebar.radio("Select viscosity equation for the plot:", ('Adam-Gibbs', 
                                                                      'Vogel-Fulcher-Tammann (VFT)', 
                                                                      'MYEGA', 
                                                                      'Avramov-Milchev', 
                                                                      'Free Volume'))

# Function to normalize composition
def normalize_composition(composition_dict):
    total = sum(composition_dict.values())
    return {oxide: value / total * 100 for oxide, value in composition_dict.items()}

#function to prepare input data for the model
@st.cache_data
def prepare_input_data(normalized_composition, composition_type):

    Inputs_ = imelt.generate_query_single(
        sio2=normalized_composition['SiO2'], 
        al2o3=normalized_composition['Al2O3'], 
        na2o=normalized_composition['Na2O'], 
        k2o=normalized_composition['K2O'],
        mgo=normalized_composition['MgO'], 
        cao=normalized_composition['CaO'],
        composition_mole=(composition_type == "mol%")
    )

    return Inputs_

# Calculate button
if st.button('Calculate Properties'):

    # PROPERTIES
    n_sample = 50

    methods = ["tg", 
               "density_glass", 
               "sellmeier", 
               "sctg", 
               "fragility", 
               "liquidus",
               "elastic_modulus", 
               "cte", 
               "aCpl", 
               "ap_calc", 
               "bCpl", 
               "a_tvf", 
               "b_tvf", 
               "c_tvf",
               "ae", 
               "a_am", 
               "a_cg", 
               "be", 
               "b_cg", 
               "to_cg", 
               "c_cg", 
               "raman_spectra"]

    # Prepare the input data
    normalized_composition = normalize_composition(composition)

    composition_prepared = prepare_input_data(normalized_composition, composition_type)

    predicted_properties = neuralmodel.predict(methods, 
                                            composition_prepared, 
                                            lbd = np.array([589.0*1e-3,]), 
                                            sampling=True, 
                                            n_sample=n_sample)

    tg = predicted_properties["tg"]
    density = predicted_properties["density_glass"]
    ri = predicted_properties["sellmeier"]
    sctg = predicted_properties["sctg"]
    fragility = predicted_properties["fragility"]
    liquidus = predicted_properties["liquidus"]
    elastic = predicted_properties["elastic_modulus"]
    cte = predicted_properties["cte"]

    aCpl = predicted_properties["aCpl"]
    ap = predicted_properties["ap_calc"]
    bCpl = predicted_properties["bCpl"]

    # TVF parameters
    A_VFT = predicted_properties["a_tvf"]
    B_VFT = predicted_properties["b_tvf"]
    C_VFT = predicted_properties["c_tvf"]

    # parameters for other viscosity models
    A_AG = predicted_properties["ae"]
    A_AM = predicted_properties["a_am"]
    A_CG = predicted_properties["a_cg"]

    B_AG = predicted_properties["be"]
    B_CG = predicted_properties["b_cg"]

    TO_CG = predicted_properties["to_cg"]
    C_CG = predicted_properties["c_cg"]

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
        
    st.subheader('Predicted latent and observed properties')

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
        
    st.subheader('Viscosity equations and parameters')

    st.markdown("Vogel-Fulcher-Tammann")
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

    correspondance = {"Adam-Gibbs": "ag", "Avramov-Milchev": "am", "Free Volume": "cg", "Vogel-Fulcher-Tammann (VFT)": "tvf", "MYEGA": "myega"}
    labels_ = ["Temperature, K", correspondance[equation_]]

    # Viscosity prediction with the choosen equation
    viscosity = neuralmodel.predict(correspondance[equation_],
                                    composition_prepared,
                                    T_range.reshape(-1,1), sampling=True, n_sample=n_sample)

    visco_mean = viscosity.mean(axis=1)
    visco_lb = visco_mean - viscosity.std(axis=1)
    visco_ub = visco_mean + viscosity.std(axis=1)

    # record in dataframe for output
    viscosity_dataframe = pd.DataFrame(np.vstack((T_range.ravel(),visco_mean)).T, 
                                                columns=["Temperature, K", equation_])

    # Raman spectra predictions
    raman_shift=np.arange(400.,1250.,1.0)
    raman = predicted_properties["raman_spectra"]

    # calculate mean values and boundaries
    raman_mean = raman.mean(axis=2).ravel()
    raman_lb = raman_mean - raman.std(axis=2).ravel()
    raman_ub = raman_mean + raman.std(axis=2).ravel()

    # record in dataframe for output
    raman_spectra = pd.DataFrame(np.vstack((raman_shift.ravel(), raman_mean)).T, 
                                columns=["Raman shift, cm-1","Intensity"])

    #
    # Make figures
    #
    st.subheader('Figures of melt viscosity and glass Raman spectra')
    # VISCOSITY
    fig2 = go.Figure([
        go.Scatter(x=T_range, y=visco_mean, 
                line=dict(color='rgb(0,0,0)'), mode='lines',
                name=equation_, showlegend=True),
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

    # Display normalized composition
    st.subheader('Normalized Composition')
    st.write(pd.DataFrame([normalized_composition]).T.rename(columns={0: f'Normalized {composition_type}'}))