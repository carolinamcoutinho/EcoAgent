import streamlit as st
import pandas as pd
import model_predictor 
import numpy as np 
import joblib
import matplotlib.pyplot as plt
import agente_analyzer 

st.set_page_config(
    page_title="EcoAgent – Previsão Inteligente de Consumo",
    layout="wide"
)

st.title("🔌 EcoAgent – Previsão Inteligente de Consumo")
st.write("Preencha as informações abaixo para estimar o consumo energético e receber análises inteligentes personalizadas.")

@st.cache_resource
def get_agent():
    """Inicializa o agente inteligente (cache para performance)."""
    return agente_analyzer.EnergyAnalysisAgent()

agent = get_agent()

model = model_predictor.model
scaler = model_predictor.scaler
COLUMNS_ORDEM_FINAL = model_predictor.COLUMNS_ORDEM_FINAL

if model is None:
    st.error("❌ Falha ao carregar o modelo de predição. Verifique os caminhos no `model_predictor.py`.")

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

HVACUsage = 1 if HVACUsage_input == "On" else 0
LightingUsage = 1 if LightingUsage_input == "On" else 0
Holiday = 1 if Holiday_input == "Yes" else 0

day_map = {
    "Friday":"Day_Friday", "Monday":"Day_Monday", "Saturday":"Day_Saturday", "Sunday":"Day_Sunday", 
    "Thursday":"Day_Thursday", "Tuesday":"Day_Tuesday", "Wednesday":"Day_Wednesday"
}

days_one_hot = {d: 0 for d in day_map.values()}
days_one_hot[day_map[DayOfWeek]] = 1

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

df_input_viz = pd.DataFrame([input_dict])
try:
    cols_for_viz = [c for c in COLUMNS_ORDEM_FINAL if c in df_input_viz.columns]
    st.subheader("📄 Pré-visualização dos dados enviados para o modelo")
    st.dataframe(df_input_viz[cols_for_viz].T, width=600)
except Exception:
    st.subheader("📄 Pré-visualização dos dados enviados para o modelo (Ordem Padrão)")
    st.dataframe(df_input_viz.T, width=600)

if st.button("🔮 Realizar Previsão", type="primary"):
    
    if model is None:
        st.error("O modelo não está disponível para predição.")
    else:
        try:
            pred = model_predictor.predict_energy_consumption(input_dict)

            coef_series = pd.Series(model.coef_.flatten(), index=COLUMNS_ORDEM_FINAL)
            coef_series = coef_series.sort_values(key=abs, ascending=False)
            
            analysis = agent.analyze_consumption(
                input_data=input_dict,
                coefficients=coef_series,
                prediction=pred,
                top_n=5
            )
            
            st.markdown("---")
            st.subheader("🎯 Resultado da Previsão")
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("⚡ Consumo Estimado", f"{pred:.2f} kWh")
            with col_metric2:
                consumption_level = analysis['consumption_level']
                st.metric("📊 Nível de Consumo", consumption_level['label'])
            
            st.markdown(analysis['summary'])
            
            st.markdown("---")
            st.subheader("🔍 Análise Detalhada dos Principais Fatores")
            
            for i, explanation in enumerate(analysis['explanations'], 1):
                if explanation and explanation['message']:
                    with st.expander(f"**{i}. {explanation['feature']}** (Impacto: {explanation['impact']})", expanded=(i==1)):
                        st.markdown(explanation['message'])
                        
                        coef_val = explanation['coefficient']
                        st.caption(f"📈 Coeficiente do modelo: {coef_val:.4f}")
            
            st.markdown("---")
            st.subheader("📊 Importância das Variáveis")
            st.write("Valores **positivos** aumentam o consumo previsto; valores **negativos** reduzem.")
            
            coef_series_viz = coef_series.copy()
            coef_series_viz.index = coef_series_viz.index.str.replace('Day_', 'Dia ')
            
            st.bar_chart(coef_series_viz.head(10))
            

            st.markdown("---")
            st.subheader("💡 Recomendações Personalizadas")
            
            for i, recommendation in enumerate(analysis['recommendations'], 1):
                st.markdown(f"{i}. {recommendation}")
            

            st.markdown("---")
            st.subheader("📈 Comparação de Cenários")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.info("**Cenário Atual**")
                st.write(f"Consumo: **{pred:.2f} kWh**")
                
            with col_comp2:
                optimized_input = input_dict.copy()
                
                if input_dict.get('Temperature', 0) > 28:
                    optimized_input['Temperature'] = 26
                if input_dict.get('Temperature', 0) < 18:
                    optimized_input['Temperature'] = 20
                    
                try:
                    pred_optimized = model_predictor.predict_energy_consumption(optimized_input)
                    economia = pred - pred_optimized
                    percentual = (economia / pred * 100) if pred > 0 else 0
                    
                    st.success("**Cenário Otimizado**")
                    st.write(f"Consumo: **{pred_optimized:.2f} kWh**")
                    if economia > 0:
                        st.write(f"💰 Economia potencial: **{economia:.2f} kWh** ({percentual:.1f}%)")
                except:
                    st.write("Otimização não disponível")

        except Exception as e:
            st.error("❌ Erro ao realizar a predição ou análise.")
            st.write(f"Detalhes: {e}")
            import traceback
            st.code(traceback.format_exc())

st.write("---")
st.write("📍 **EcoAgent** – Sistema Inteligente para Otimização de Consumo Energético.")
st.caption("Desenvolvido com IA para análises contextualizadas e recomendações personalizadas.")