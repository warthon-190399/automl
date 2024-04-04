import pandas as pd
import streamlit as st
from eda import exploratory_data_analysis
from modelos import ml_analysis

st.set_page_config(page_title="Auto-ML Analyzer",
                   page_icon=":robot_face:",
                   initial_sidebar_state="expanded",
                   layout="centered"
                   )

def handle_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    dummie_if = st.sidebar.radio("¿Cuenta con variables categóricas a codificar?", ("Si", "No"))

    if dummie_if == "Si":
        dummie_var = st.sidebar.multiselect("Variables categóricas:", categorical_columns)

        if not dummie_var:
            with st.empty():
                st.info("Tu modelo no presenta variables categóricas a codificar. Por favor, indica 'No'.")
                st.info("Asegúrate de seleccionar todas las variables categóricas antes de continuar.")
        else:
            df_dummies = pd.get_dummies(df[dummie_var], prefix=dummie_var)
            df = pd.concat([df, df_dummies], axis=1)
            df = df.drop(dummie_var, axis=1)
    else:
        with st.empty():
            st.info("Aún quedan más variables por codificar. Por favor, indica 'Si' para continuar.")

    return df

st.title("¡Bienvenido a Auto-ML Analyzer!")
st.write("En Auto-ML Analyzer, encontrarás una plataforma completa para explorar tus datos y desarrollar modelos de machine learning de manera eficiente.")
st.write("En la sección de Análisis Exploratorio de Datos, podrás explorar gráficos interactivos que te permitirán visualizar tus variables y comprender mejor su distribución y relación entre sí.")
st.write("En la sección de Análisis de Modelos, podrás seleccionar entre modelos de regresión y clasificación. Cada uno de estos modelos viene con una serie de opciones que te permitirán evaluar su desempeño y determinar cuál se ajusta mejor a tus datos.")
st.sidebar.title("Auto Machine Learning Analyzer")
df = st.file_uploader(label="Por favor, carga tu archivo csv:")
section = st.sidebar.radio("Sección:", ("Análisis Exploratorio de Datos", "Análisis de Modelos de Machine Learning"))

if df:
    df = pd.read_csv(df)
    df=handle_categorical_variables(df)
    df.dropna()
    st.write(df.head())

    if section == "Análisis Exploratorio de Datos":
        exploratory_data_analysis(df)
    elif section == "Análisis de Modelos de Machine Learning":
        ml_analysis(df)







