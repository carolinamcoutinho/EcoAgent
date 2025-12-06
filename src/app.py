import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# --------------------------------------------------
st.set_page_config(
    page_title="EcoAgent – Previsão Inteligente de Consumo",
    layout="wide"
)

st.title("🔌 EcoAgent – Previsão Inteligente de Consumo")
st.write("Preencha as informações abaixo para estimar o consumo energético e visualizar os fatores que mais influenciaram a predição.")

# --------------------------------------------------
# CARREGAR MODELO E SCALER
# --------------------------------------------------
import os
import joblib

# Caminho absoluto baseado na localização do arquivo app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "ecoagent_linear_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "ecoagent_scaler.pkl")

# Normaliza o caminho (para evitar erros de ../ em Windows)
MODEL_PATH = os.path.normpath(MODEL_PATH)
SCALER_PATH = os.path.normpath(SCALER_PATH)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)

    try:
        scaler = joblib.load(SCALER_PATH)
    except:
        scaler = None

    return model, scaler

model, scaler = load_artifacts()


# ==================================================
# SEÇÃO DE INPUTS
# ==================================================
st.header("📥 Insira os dados")

col1, col2, col3 = st.columns(3)

with col1:
    Temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=-10.0, max_value=50.0, value=25.0,
        help="Temperatura ambiente atual em graus Celsius."
    )

    Humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0, max_value=100.0, value=50.0,
        help="Umidade relativa do ar como porcentagem."
    )

    SquareFootage = st.number_input(
        "🏠 Square Footage (ft²)",
        min_value=100, max_value=10000, value=1200,
        help="Tamanho do ambiente em pés quadrados."
    )

with col2:
    Occupancy = st.number_input(
        "👥 Occupancy (nº de pessoas)",
        min_value=0, max_value=20, value=2,
        help="Quantidade de pessoas presentes no ambiente."
    )

    HVACUsage_input = st.selectbox(
        "❄️ HVAC Usage",
        ["Off", "On"],
        help="Indique se o sistema de climatização está ligado."
    )

    LightingUsage_input = st.selectbox(
        "💡 Lighting Usage",
        ["Off", "On"],
        help="Indique se a iluminação principal está ligada."
    )

with col3:
    RenewableEnergy = st.number_input(
        "🔆 Renewable Energy (%)",
        min_value=0.0, max_value=100.0, value=20.0,
        help="Percentual do consumo vindo de energia renovável."
    )

    DayOfWeek = st.selectbox(
        "📅 Day of the Week",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        help="Dia da semana referente à medição."
    )

    Holiday_input = st.selectbox(
        "🎉 Holiday",
        ["No", "Yes"],
        help="Indique se o dia é um feriado."
    )


# ==================================================
# CONVERSÕES PARA FORMATO USADO PELO MODELO
# ==================================================

HVACUsage = 1 if HVACUsage_input == "On" else 0
LightingUsage = 1 if LightingUsage_input == "On" else 0
Holiday = 1 if Holiday_input == "Yes" else 0

day_map = {
    "Friday":"Day_Friday",
    "Monday":"Day_Monday",
    "Saturday":"Day_Saturday",
    "Sunday":"Day_Sunday",
    "Thursday":"Day_Thursday",
    "Tuesday":"Day_Tuesday",
    "Wednesday":"Day_Wednesday"
}

days_one_hot = {d: 0 for d in day_map.values()}
days_one_hot[day_map[DayOfWeek]] = 1

# ==================================================
# MONTAR DATAFRAME FINAL
# ==================================================
input_dict = {
    "Temperature": Temperature,
    "Humidity": Humidity,
    "SquareFootage": SquareFootage,
    "Occupancy": Occupancy,
    "HVACUsage": HVACUsage,
    "LightingUsage": LightingUsage,
    "RenewableEnergy": RenewableEnergy,
    "Holiday": Holiday
}

input_dict.update(days_one_hot)

df_input = pd.DataFrame([input_dict])

st.subheader("📄 Pré-visualização dos dados enviados para o modelo")
st.dataframe(df_input.T, width=600)


# ==================================================
# BOTÃO DE PREDIÇÃO
# ==================================================
if st.button("🔍 Realizar Previsão"):

    X = df_input.copy()

    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            X_model = X_scaled
        except:
            st.error("Erro ao aplicar scaler. Verifique as colunas.")
            X_model = X.values
    else:
        X_model = X.values

    pred = model.predict(X_model)[0]

    st.subheader("🔮 Resultado da Previsão")
    st.metric("Consumo Estimado (kWh)", f"{pred:.2f}")

    # ==================================================
    # EXPLICAÇÃO USANDO APENAS COEFICIENTES
    # ==================================================
    st.subheader("📊 Variáveis que mais influenciaram a predição")

    try:
        coef_series = pd.Series(model.coef_.flatten(), index=df_input.columns)
        coef_series = coef_series.sort_values(key=abs, ascending=False)

        st.write("Valores positivos aumentam o consumo previsto; valores negativos reduzem.")

        st.bar_chart(coef_series)

    except Exception as e:
        st.error("Não foi possível calcular a importância das variáveis.")
        st.write(e)

st.write("---")
st.write("🔍 **EcoAgent** — Sistema Inteligente para Otimização de Consumo Energético.")
