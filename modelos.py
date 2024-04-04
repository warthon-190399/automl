from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def ml_analysis(df):    

    st.sidebar.subheader("Bienvenido a la sección de Análisis de Modelos.")
    st.sidebar.write("Selecciona las opciones para el análisis de modelo:")

    variable_depen = st.sidebar.selectbox("Variable Dependiente:", df.columns.tolist())
    y = df[variable_depen]
    x = df.drop(columns=variable_depen)

   # División de los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0
                                                        )

    # Imputación de valores faltantes
    imputer = SimpleImputer(strategy='mean')  # Utiliza 'mean' para llenar los valores faltantes con la media
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    #Agregar estadarizacion y normalizacion

    dimension_tipo = st.sidebar.radio("Selección de Escalado", ("Estandarización", "Normalización"))
    if dimension_tipo == "Estandarización":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif dimension_tipo == "Normalización":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    modelo_tipo = st.sidebar.radio("Tipo de Modelo", ("Regresión", "Clasificación"))

    if modelo_tipo == "Regresión":
        modelos_reg = {
            "Linear Regression": LinearRegression(),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
            "Support Vector Machine": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
        }

        modelo_selec = st.sidebar.selectbox("Selecciona el modelo de regresión", list(modelos_reg.keys()))

        modelo = modelos_reg[modelo_selec]
        modelo.fit(x_train, y_train)
        y_pred = modelo.predict(x_test)

        # Evaluación
        st.subheader(f"Métricas de Evaluación de {modelo_selec}")

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Mean Squared Error:", mse)
        st.write("Mean Absolute Error:", mae)
        st.write("R^2 Score:", r2)

    #Comparacion valores reales vs predichos
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Scatter(x=y_test,
                                            y=y_pred,
                                            mode='markers',
                                            name='Valores Predichos vs. Reales'
                                            )
                                )
        fig_comparison.add_trace(go.Scatter(x=y_test,
                                            y=y_test,
                                            mode='lines',
                                            name='Línea de Referencia'
                                            )
                                )
        fig_comparison.update_layout(title="Comparación entre Valores Reales y Predichos",
                                    xaxis_title="Valores Reales",
                                    yaxis_title="Valores Predichos",
                                    legend=dict(orientation="h",
                                                yanchor="bottom",
                                                y=-0.5,
                                                xanchor="right",
                                                x=1
                                                )
                                    )

        #Distribucion de errores
        errors = y_test - y_pred
        fig_error_distribution = go.Figure(data=[go.Histogram(x=errors, histnorm='probability')])
        fig_error_distribution.update_layout(title="Distribución de Errores", xaxis_title="Error", yaxis_title="Frecuencia")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_error_distribution, use_container_width=True)

    if modelo_tipo == "Clasificación":
        if len(y.unique()) > 2:
            st.error("La variable dependiente seleccionada tiene más de dos clases. Asegúrate de seleccionar una variable con solo dos clases para problemas de clasificación binaria.")
        
        else:
            modelos_clas = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=0),
                "Support Vector Classifier": SVC(kernel="rbf", probability=True,random_state=0),
                "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, random_state=0),
                "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=0),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=0),
                "Naive Bayes": GaussianNB(),
                "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
                "Kernel SVM": SVC(kernel="linear", probability=True, random_state=0)
            }

            modelo_selec = st.sidebar.selectbox("Selecciona el modelo de clasificación", list(modelos_clas.keys()))

            modelo = modelos_clas[modelo_selec]
            modelo.fit(x_train, y_train)
            y_pred = modelo.predict(x_test)

            #Evaluacion
            st.subheader(f"Métricas de Evaluación de {modelo_selec}")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Curva ROC
            prob = modelo.predict_proba(x_test)
            fpr, tpr, _ = roc_curve(y_test, prob[:,1])
            auc = roc_auc_score(y_test, prob[:, 1])

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr,
                                        y=tpr,
                                        mode="lines",
                                        name="ROC Curve"
                                        )
                            )
            fig_roc.add_shape(type="line",
                            line=dict(color="yellow",dash="dash"),
                            x0=0,
                            x1=1,
                            y0=0,
                            y1=1
                            )
            fig_roc.update_layout(title=f'Curva ROC ({modelo_selec})',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate'
                                )
            
            #matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm,
                            labels=dict(x="Predicción", y="Verdadero"),
                            x=["Negativo", "Positivo"],
                            y=["Negativo", "Positivo"]
                            )
            fig_cm.update_layout(title=f'Matriz de Confusión ({modelo_selec})')

            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("Accuracy:", accuracy)
                st.write("Precision:", precision)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                st.write("Recall:", recall)
                st.write("F1 Score:", f1)
                st.write("")
                st.plotly_chart(fig_cm, use_container_width=True)
    









































