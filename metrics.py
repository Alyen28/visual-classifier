import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from collections import defaultdict
from sklearn.metrics import confusion_matrix
from translations import get_text, get_class_name


@st.cache_data
def calcular_acuracia_por_classe(y_test, y_pred_test, idioma):
    """
    Calcula a acurácia individual de cada classe.
    """
    acertos_por_classe = defaultdict(int)
    total_por_classe = defaultdict(int)

    for classe_real, classe_prevista in zip(y_test, y_pred_test):
        total_por_classe[classe_real] += 1

        if classe_real == classe_prevista:
            acertos_por_classe[classe_real] += 1

    resultados = []

    for classe_id in sorted(total_por_classe.keys()):
        acuracia = acertos_por_classe[classe_id] / total_por_classe[classe_id] * 100

        resultados.append({
            "class_id": classe_id,
            "class_name": get_class_name(classe_id, idioma),
            "accuracy": round(acuracia, 1)
        })

    return pd.DataFrame(resultados)


@st.cache_data
def gerar_dataframe_matriz(y_true, y_pred, idioma):
    """
    Gera a matriz de confusão em formato de dataframe.
    """
    matriz = confusion_matrix(y_true, y_pred)
    nomes_classes = [get_class_name(i, idioma) for i in range(10)]

    matriz_df = pd.DataFrame(
        matriz,
        index=nomes_classes,
        columns=nomes_classes
    )

    return matriz_df


@st.cache_data
def gerar_grafico_matriz(matriz_df, idioma):
    """
    Gera um gráfico da matriz de confusão.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    imagem = ax.imshow(matriz_df.values, cmap="Blues")

    ax.set_title(get_text(idioma, "confusion_matrix_title"), fontsize=15, pad=14)
    ax.set_xlabel(get_text(idioma, "predicted_class"))
    ax.set_ylabel(get_text(idioma, "true_class"))

    ax.set_xticks(np.arange(len(matriz_df.columns)))
    ax.set_yticks(np.arange(len(matriz_df.index)))

    ax.set_xticklabels(matriz_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(matriz_df.index)

    for linha in range(matriz_df.shape[0]):
        for coluna in range(matriz_df.shape[1]):
            valor = matriz_df.iloc[linha, coluna]
            ax.text(coluna, linha, str(valor), ha="center", va="center", fontsize=8)

    fig.colorbar(imagem, ax=ax)
    fig.tight_layout()

    return fig