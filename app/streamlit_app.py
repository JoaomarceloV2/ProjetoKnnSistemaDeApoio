import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Previsor ENEM KNN", page_icon="", layout="centered")

st.title("Previsão de Nota ENEM com KNN ")


TARGET = "NU_NOTA_MT"
MODEL_PATH = Path("models") / f"knn_{TARGET}.joblib"
META_PATH = Path("models") / f"knn_{TARGET}_meta.json"
SAMPLE_SIZE = 50_000
DATA_PATH = Path("data/processed") / f"enem_2023_amostra_{SAMPLE_SIZE}.parquet"

@st.cache_resource
def load_artifacts():
    """ Carrega o modelo, metadados e os dados de amostra uma única vez. """
    try:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        df_sample = pd.read_parquet(DATA_PATH)
        return model, meta, df_sample
    except FileNotFoundError:
        st.error(
            f"Erro: Arquivo do modelo ou de metadados não encontrado. "
            f"Por favor, execute o script de treinamento 'src/02_train.py' primeiro."
        )
        return None, None, None

model, meta, df_base = load_artifacts()


if model is None or meta is None or df_base is None:
    st.stop()
    
CAT_FEATURES = meta["features"]["categorical"]
NUM_FEATURES = meta["features"]["numeric"]

with st.form("enem_form"):
    st.subheader("Insira os dados do candidato")
    
    q006_map = {'A': 'Nenhuma renda', 'B': 'Até R$ 1.412,00', 'C': 'De R$ 1.412,01 a R$ 2.118,00', 'D': 'De R$ 2.118,01 a R$ 2.824,00', 'E': 'De R$ 2.824,01 a R$ 3.530,00', 'F': 'De R$ 3.530,01 a R$ 4.236,00', 'G': 'De R$ 4.236,01 a R$ 5.648,00', 'H': 'De R$ 5.648,01 a R$ 7.060,00', 'I': 'De R$ 7.060,01 a R$ 8.472,00', 'J': 'De R$ 8.472,01 a R$ 9.884,00', 'K': 'De R$ 9.884,01 a R$ 11.296,00', 'L': 'De R$ 11.296,01 a R$ 14.120,00', 'M': 'De R$ 14.120,01 a R$ 16.944,00', 'N': 'De R$ 16.944,01 a R$ 19.768,00', 'O': 'De R$ 19.768,01 a R$ 28.240,00', 'P': 'Acima de R$ 28.240,00', 'Q': 'Não sei'}
    tp_escola_map = {1: 'Não Respondeu', 2: 'Pública', 3: 'Privada'}

    col1, col2 = st.columns(2)
    with col1:
        Q006 = st.selectbox("Renda Familiar (Q006)", options=q006_map.keys(), format_func=lambda x: q006_map.get(x, "N/A"))
        TP_ESCOLA = st.selectbox("Tipo de Escola (TP_ESCOLA)", options=tp_escola_map.keys(), format_func=lambda x: tp_escola_map.get(x, "N/A"))
        
    with col2:
        Q002 = st.selectbox("Escolaridade da Mãe (Q002)", sorted(df_base["Q002"].dropna().unique()))
        TP_COR_RACA = st.selectbox("Cor/Raça (TP_COR_RACA)", sorted(df_base["TP_COR_RACA"].dropna().unique()))

    st.subheader("Notas conhecidas (deixe 0 se não souber)")
    col_notas1, col_notas2 = st.columns(2)
    with col_notas1:
        NU_NOTA_CN = st.number_input("Nota em Ciências da Natureza", min_value=0.0, max_value=1000.0, value=500.0)
        NU_NOTA_LC = st.number_input("Nota em Linguagens e Códigos", min_value=0.0, max_value=1000.0, value=500.0)
    with col_notas2:
        NU_NOTA_CH = st.number_input("Nota em Ciências Humanas", min_value=0.0, max_value=1000.0, value=500.0)
        NU_NOTA_RED = st.number_input("Nota da Redação", min_value=0.0, max_value=1000.0, value=500.0)
        
    submitted = st.form_submit_button(f"Prever nota em {TARGET}")

if submitted:
    input_data = {
        "Q006": Q006, "Q002": Q002, "TP_ESCOLA": TP_ESCOLA, "TP_COR_RACA": TP_COR_RACA, 
        "NU_NOTA_CN": NU_NOTA_CN, "NU_NOTA_LC": NU_NOTA_LC, "NU_NOTA_CH": NU_NOTA_CH, 
        "NU_NOTA_RED": NU_NOTA_RED
    }
    
    for feature in NUM_FEATURES:
        if input_data[feature] == 0.0:
            input_data[feature] = np.nan
            
    X_input = pd.DataFrame([input_data])
    
    prediction = model.predict(X_input)[0]
    
    st.success(f"**Nota prevista em Matemática: {prediction:.2f}**")
    
    st.subheader("Análise de Comparação com Perfis Similares")
    
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    X_input_transformed = preprocessor.transform(X_input)
    
    distances, indices = regressor.kneighbors(X_input_transformed)
    
    df_neighbors = df_base.iloc[indices[0]]
    
    st.write("Os 5 perfis mais parecidos com o candidato informado na base de dados são:")
    
    display_cols = [TARGET] + NUM_FEATURES
    st.dataframe(df_neighbors[display_cols].style.format("{:.2f}"))
    
    mean_neighbors = df_neighbors[TARGET].mean()
    
    if prediction > mean_neighbors:
        st.info(f"A nota prevista de **{prediction:.2f}** está **acima** da média dos perfis similares ({mean_neighbors:.2f}).")
    else:
        st.info(f"A nota prevista de **{prediction:.2f}** está **abaixo** da média dos perfis similares ({mean_neighbors:.2f}).")