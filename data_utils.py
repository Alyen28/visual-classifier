import joblib
import pandas as pd
import streamlit as st

@st.cache_data
def carregar_dados_teste():
    """
    Carrega o conjunto de teste do Fashion-MNIST.
    """
    test_data = pd.read_csv("data/fashion-mnist_test.csv")
    return test_data


@st.cache_resource
def carregar_modelo(caminho_modelo):
    """
    Carrega os artefatos do modelo treinado pelo arquivo train_model.py.
    """
    artifacts = joblib.load(caminho_modelo)
    return artifacts