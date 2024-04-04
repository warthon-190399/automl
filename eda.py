#%%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Inicializar el estado de la sesión en eda.py


def exploratory_data_analysis(df):
    st.sidebar.subheader("Bienvenido al Análisis Exploratorio de Datos automatizado de AutoML.")
    x_var = st.sidebar.selectbox("Variable X:", df.columns)
    y_var = st.sidebar.selectbox("Variable Y:", df.columns)
    #%%
    matriz_corr = df.corr()
    fig_matriz = px.imshow(matriz_corr,
                        color_continuous_scale='Viridis',
                        height=500,
                        template="plotly_dark"
                        )
    fig_matriz.update_layout(
        margin={'r': 0, 't': 0, 'b': 0, 'l': 0}
    )


    #%%
    fig_hist = px.histogram(df,
                            x=x_var,
                            nbins=15,
                            barmode='stack',
                            marginal="rug",
                            height=400,
                            width=400,
                            )

    # %%
    fig_box = px.box(df,
                    y=x_var,
                    template="plotly_dark",
                    points="all",
                    height=400,
                    width=400,
                    )
    # %%
    fig_violin = px.violin(df,
                        y=x_var,
                        box=True,
                        orientation="v",
                        template="plotly_dark",
                        height=400,
                        width=400,
                        )

    # %%
    fig_regresion = px.scatter(df,
                            x=x_var,
                            y=y_var,
                            trendline='ols',
                            template="plotly_dark"
                            )
    # %%
    st.subheader("Matriz de Correlaciones")
    st.plotly_chart(fig_matriz, use_container_width=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Distribución de Datos de f{x_var}")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.subheader(f"Boxplot de {x_var}")
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        st.subheader(f"Violinplot de {x_var}")
        st.plotly_chart(fig_violin, use_container_width=True)
        st.subheader(f"Regresión de {x_var} y {y_var}")
        st.plotly_chart(fig_regresion, use_container_width=True)















