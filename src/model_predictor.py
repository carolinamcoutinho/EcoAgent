import pandas as pd
import joblib
import os
from typing import Dict, Any
import numpy as np # Adicionada importação de numpy

# --- CONFIGURAÇÃO DE CAMINHOS ---
# Usamos o caminho relativo ao arquivo predictor para encontrar os modelos
# ATENÇÃO: Se o predictor está em um nível diferente do diretório 'models', este caminho pode precisar ser ajustado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Assumindo que 'models' está na mesma pasta que 'model_predictor.py'
MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.pkl") 
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# CRUCIAL: Ordem EXATA das features usada no treino do modelo
COLUMNS_ORDEM_FINAL = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy',
    'HVACUsage', 'LightingUsage', 'Holiday', 
    'Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday', 
    'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday'
]
# Colunas que DEVEM ser escaladas (estas são as colunas que o scaler foi ajustado)
COLS_TO_SCALE = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']


# --- CARREGAMENTO DE ARTEFATOS ---

def load_artifacts():
    """Carrega o modelo e o scaler do disco."""
    try:
        model = joblib.load(MODEL_PATH)
        # O scaler do scikit-learn geralmente guarda a lista de features que usou para o fit.
        # É uma boa prática usar essa lista para garantir a ordem correta, mas
        # para este exemplo, dependemos do COLUMNS_ORDEM_FINAL.
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError as e:
        # Permite que o app rode mesmo se um arquivo estiver faltando (para o Streamlit)
        print(f"Aviso: Arquivo não encontrado: {e.filename}. Retornando artefatos parciais.")
        return None, None # Retorna None para ambos ou lida com a exceção de outra forma
    except Exception as e:
        print(f"Erro inesperado ao carregar artefatos: {e}")
        return None, None

# Os artefatos são carregados na primeira importação
model, scaler = load_artifacts()

if model is not None:
    print("✅ Artefatos de ML carregados com sucesso no predictor.")
else:
    print("❌ Falha ao carregar artefatos. Verifique os caminhos.")


# --- FUNÇÃO PRINCIPAL DE PREDIÇÃO ---

def predict_energy_consumption(input_data: Dict[str, Any]) -> float:
    """
    Recebe os dados de entrada (já convertidos), aplica o pré-processamento 
    (ordem e scaling) e retorna a previsão de consumo.

    Args:
        input_data: Dicionário com todos os dados do usuário.

    Returns:
        O consumo de energia previsto em kWh.
    """
    if model is None:
        raise RuntimeError("O modelo de ML não foi carregado corretamente.")

    # 1. Converter para DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 2. Alinhar e reordenar as colunas
    # Isso garante que a ordem das colunas seja exatamente a esperada pelo modelo
    try:
        X = df_input[COLUMNS_ORDEM_FINAL].copy()
    except KeyError as e:
        # Caso alguma coluna esperada esteja faltando no input_data
        raise KeyError(f"Dados de entrada faltando a feature: {e}. A ordem esperada é: {COLUMNS_ORDEM_FINAL}")
    
    X_model = X.copy()
    
    # 3. Aplicar o Scaler
    if scaler is not None:
        try:
            # Seleciona APENAS as colunas que DEVEM ser transformadas (COLS_TO_SCALE)
            X_scaled_part = X[COLS_TO_SCALE]
            
            # Aplica a transformação e substitui NO DataFrame modelo
            X_model[COLS_TO_SCALE] = scaler.transform(X_scaled_part)
        except Exception as e:
            # Se houver erro no scaling, o ideal é FALHAR, mas se quiser continuar:
            print(f"Aviso: Erro crítico ao aplicar scaler: {e}. Usando dados brutos.")
            X_model = X.copy() # Usa os dados sem scaling

    # 4. Realizar a Predição
    # A função predict do scikit-learn geralmente aceita DataFrames ou arrays.
    # Usar .values é mais seguro, mas reshape(1, -1) é necessário para garantir 2D.
    pred = model.predict(X_model.values.reshape(1, -1))[0]
    
    return pred