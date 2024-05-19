#
# Feux de forêts - Canada - 1990 à 2021
# 2025-05
# Jean-Pierre Tanguay
#
#
# 
#
import plotly.express as px

import streamlit as st
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from scipy.fft import fft, ifft, fftfreq 

import os

#st.write(os.getcwd())

st.set_page_config(
    page_title="Feux de forêts", 
    page_icon=None, 
    layout="wide", 
    initial_sidebar_state="expanded", 
    menu_items=None
)

@st.cache_data
def load_data():
    df = pd.read_csv("./wildfire-canada-1990-2021.csv")
    return df



#######################################
# Load and process data
#######################################

df = load_data()
#df

colnames = df.columns.to_list()



#######################################
#  display
#######################################


st.title("Feux de forêts - Canada - 1990 à 2021")




#col2.image("countries.png")
st.write(
    """
    Environnement Canada
    
    Les superficies sont en hectares
    """
    
)




st.header("Données source")

jurisdictions = np.sort(df.jurisdiction.unique())
origins = np.sort(df.origin.unique()) 

with st.container(border=True):

    #col1, col2, col3 = st.columns([2,1,1])
    
    tab1, tab2, tab3 = st.tabs(["Données", "Statistiques", "Catégories"])


    with tab1:
        st.write(df)
             
        st.write(
            "<a href='/' target='_blank'>Data source</a>",
            unsafe_allow_html=True
        )   
    #with st.expander("Statistiques"):
        
    with tab2:
        col1, col2 = st.columns([3,4] )
        with col1:
            st.subheader("Numeriques")
            st.write( df.describe() )
        with col2:
            st.subheader("Catégories")
            st.write( df.describe(include = 'object') )





    with tab3: #st.container(border=True):

        col1, col2 = st.columns(2)
        with col1:   
            "Juridiction"
            jurisdictions
        with col2:  
            "Origine"
            origins
        
 

############################################
# superficie
############################################ 
       
date_start = int(df["year"].min())
date_end = int(df["year"].max())

#date_start
#date_end
    
df_2 = df
#df_2

##########################################################
# sidebar
##########################
jur_list = ["Tout"]
jur_list.extend(jurisdictions)
#jur_list


sb = st.sidebar
sb.title("Juridictions")
sb.header("Provinces et territoires")
jur = sb.radio("Sélectionnez la juridiction: ", options=jur_list, horizontal = True)
#

#mask

if (jur == "Tout"):
    df_jur = df_2
else:    
    mask = (df_2["jurisdiction"] == jur)
    df_jur = df_2.loc[mask]
    

#df_jur.describe
df_jur_no_origin = df_jur.groupby([ 'year']).agg({'area': 'sum'})
df_jur_no_origin = df_jur_no_origin.reset_index()

with st.container(border=True):
    st.subheader("Superficie brulée par année"  + " (" + jur + ")")
    st.write("Sélectionnez la juridiction dans le panneau latéral")
    col1, col2, col3 = st.columns([1,1, 3] )
    with col1:
        st.write(df_jur_no_origin.describe()) 
    with col2:    
        vmin = df_jur_no_origin["area"].describe().loc["min"]
        vavg = df_jur_no_origin["area"].describe().loc["mean"]
        vmax = df_jur_no_origin["area"].describe().loc["max"]

        st.metric(":green[Min]", "%6.0f ha" % vmin, "")
        st.metric(":blue[Moyenne]", "%6.0f ha" % vavg, "")
        st.metric(":red[Max]", "%6.0f ha" % vmax, "")
        
    with col3:
        "Superficie par année"
        st.bar_chart(data = df_jur_no_origin, x = "year", y="area")

    "Histogramme des superficies"
    figy = px.histogram(df_jur, x="area")
    st.plotly_chart(figy, theme="streamlit", use_container_width=False)


with st.container(border=True):
    subh = "Origine des incendies"  + " (" + jur + ")"
    st.subheader(subh)
    
    col1, col2 = st.columns([1,3] )
    with col1:  
        df_temp = df_jur['origin'].value_counts() #.plot(kind='bar')
        "Nombre d'incendies par origine"
        df_temp        

    with col2:
        fig = px.histogram(df_jur, x="origin")
        st.plotly_chart(fig, theme="streamlit", use_container_width=False)

    "Répartitions des origines sur la période couverte"
    st.bar_chart(data = df_jur, x = "year",  y="origin")
    
    "Superficie brulée annuellement par origine"
    scale = "linear"
    import altair as alt
    c = alt.Chart(df_jur, height=500).mark_bar(size=5).encode(
        x=alt.X('year:N', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("area", axis=alt.Axis(grid=False), scale=alt.Scale(type=scale)),
        #facet="origin:O",
        xOffset="origin", #["Activités humaines", "Brûlage dirigé", "Foudre", "Indéterminée"],
        color=alt.Color('origin',
            # source: https://stackoverflow.com/questions/68624885/position-altair-legend-top-center
            legend=alt.Legend(
                orient='none',
                legendX=1000, legendY=-10,
                direction='horizontal',
                titleAnchor='middle'
            )
        )
    ).configure_header(labelOrient='bottom', labelPadding = 5).configure_facet(spacing=15)
    st.altair_chart(c, use_container_width=True)         
        
    with st.expander("Détails", expanded=False):
        col3, col4 = st.columns([2,4] )
        with col3:     
            df_jur            
        with col4: 
            df1 = pd.pivot_table(df_jur, index=['year'], columns='origin', values='area')
            df1 = df1.reset_index()
            df1


    #################
    
with st.container(border=True):    
    st.subheader("Superficie cumulée depuis 1990"  + " (" + jur + ")")

    df_cumul = df_jur[["year", "area"]] #pd.DataFrame()
    df_cumul = df_cumul.groupby(["year"]).agg({'area': 'sum'})
    df_cumul = df_cumul.cumsum(axis = 0, skipna = True)
    df_cumul = df_cumul.reset_index()
    col5, col6 = st.columns([1,4] )
    with col5:   
        df_cumul
    with col6:

        fig_cumul = px.bar(df_cumul, x="year", y="area")
        st.plotly_chart(fig_cumul, theme="streamlit", use_container_width=False)

#########################################




    
############################################
# FFT
############################################ 

# source: https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
    
with st.container(border=True):    
    st.subheader("Cycles"  + " (" + jur + ")")


    df_jur_sum = df_jur[["year", "area"]]
    df_jur_sum = df_jur_sum.groupby(["year"]).agg({'area': 'sum'})

    df_jur_sum["fft"] = np.fft.fft(df_jur_sum["area"].values)

    df_jur_sum = df_jur_sum.reset_index() 
    


    #########

    per = 1 #265*24*60*60 #(365*24* 60*60)


    X = np.fft.fft(df_jur_sum["area"])
    N = len(X)
    n = np.arange(N)
    # get the sampling rate
    sr = 1 / per
    T = N/sr
    freq = n/T 

    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]


    # convert frequency to years
    t_h = 1/f_oneside / per

    c1, c2 = st.columns(2)
    with c1:
        fig2 = plt.figure(figsize=(6, 3))
        #plt.figure(figsize = (12, 6))
        plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
        plt.xlabel('Fréq (Hz)')
        plt.ylabel('FFT Amplitude |X(fréq)|')
        plt.show()
        st.pyplot(fig2)

    with c2:
        fig2 = plt.figure(figsize=(6, 3))
        plt.plot(t_h, np.abs(X[:n_oneside])/n_oneside)
        plt.xticks([1, 2, 4, 8, 11, 16])
        plt.xlim(0, 30)
        plt.xlabel('Période ($years$)')
        plt.show()
        st.pyplot(fig2)

    with st.expander("Transformée de Fourier - détails", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            df_jur_sum
        with c2:
            X
            
            
############################################
# footer
############################################ 

st.write(  
'''
    -----------
    JP Tanguay (2024)
'''
)