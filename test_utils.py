import numpy as np


def filtrar_indices_por_classe(test_data, classe_id):
    """
    Filtra os índices do conjunto de teste com base na classe escolhida.
    """
    if classe_id == "all":
        return list(test_data.index)

    return list(test_data[test_data["label"] == classe_id].index)


def sortear_indice_sem_repetir(indices_disponiveis, historico, limite_tentativas=30):
    """
    Sorteia um índice evitando repetir imagens exibidas recentemente.
    """
    if len(indices_disponiveis) == 0:
        return None

    tentativas = 0
    indice = int(np.random.choice(indices_disponiveis))

    while indice in historico and tentativas < limite_tentativas:
        indice = int(np.random.choice(indices_disponiveis))
        tentativas += 1

    return indice


def encontrar_resultado(indices, test_data, y_pred_series, resultado_desejado):
    """
    Encontrando um exemplo aleatórioque corresponde ao filtro escolhido.
    """
    for indice in indices:
        classe_real = int(test_data.loc[indice, "label"])
        classe_prevista = int(y_pred_series.loc[indice])
        acertou = classe_real == classe_prevista

        if resultado_desejado == "correct" and acertou:
            return indice
        if resultado_desejado == "wrong" and not acertou:
            return indice

    return None


def contar_resultados(indices, test_data, y_pred_series):
    """
    Conta quantos acertos e erros existem dentro de uma lista de índices.
    """
    classes_reais = test_data.loc[indices, "label"]
    classes_previstas = y_pred_series.loc[indices]

    acertos = (classes_reais == classes_previstas).sum()
    erros = (classes_reais != classes_previstas).sum()

    return int(acertos), int(erros)