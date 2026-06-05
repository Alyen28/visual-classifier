import numpy as np
import pandas as pd
import streamlit as st

from config import MODEL_OPTIONS
from data_utils import carregar_dados_teste, carregar_modelo
from display import exibir_previsao_upload, exibir_previsao_dataset
from image_utils import preprocessar_imagem_upload
from metrics import calcular_acuracia_por_classe, gerar_grafico_matriz, gerar_dataframe_matriz
from style import aplicar_estilo
from test_utils import filtrar_indices_por_classe, sortear_indice_sem_repetir, encontrar_resultado, contar_resultados
from translations import get_text, get_class_name


def t(key):
    return get_text(st.session_state.language, key)


st.set_page_config(
    page_title="Visual Classifier",
    page_icon="🧠",
    layout="wide"
)

aplicar_estilo()

# Configura o idioma e o modelo na sidebar
if "language" not in st.session_state:
    st.session_state.language = "pt"

st.session_state.language = st.sidebar.radio(
    "Idioma / Language",
    options=["pt", "en"],
    index=0 if st.session_state.language == "pt" else 1,
    format_func=lambda lang: "Português" if lang == "pt" else "English",
    horizontal=True
)

st.sidebar.title(t("sidebar_title"))

modelo_escolhido = st.sidebar.selectbox(
    t("model_selection"),
    list(MODEL_OPTIONS.keys()),
    index=1
)

# Carrega os dados e o modelo escolhido
st.title(t("app_title"))
st.caption(t("app_description"))

with st.spinner(t("loading_page")):
    test_data = carregar_dados_teste()
    y_test = test_data["label"]
    artifacts = carregar_modelo(MODEL_OPTIONS[modelo_escolhido])

model = artifacts["model"]
y_pred_test = artifacts["y_pred_test"]
train_accuracy = artifacts["train_accuracy"]
test_accuracy = artifacts["test_accuracy"]
architecture = artifacts["architecture"]
y_pred_series = pd.Series(y_pred_test, index=test_data.index)

st.sidebar.caption(f"{t('train_accuracy')}: {train_accuracy:.1f}%")
st.sidebar.caption(f"{t('test_accuracy')}: {test_accuracy:.1f}%")

tab1, tab2 = st.tabs([t("overview_tab"), t("visual_test_tab")])

# ABA 1
with tab1:
    st.subheader(t("model_summary"))
    st.caption(f"{t('model_caption')} {modelo_escolhido}")

    col1, col2, col3 = st.columns(3)
    col1.metric(t("architecture"), architecture)
    col2.metric(t("train_accuracy"), f"{train_accuracy:.1f}%")
    col3.metric(t("test_accuracy"), f"{test_accuracy:.1f}%")

    st.subheader(t("class_accuracy"))

    accuracy_df = calcular_acuracia_por_classe(
        y_test,
        y_pred_test,
        st.session_state.language
    )

    classe_melhor = accuracy_df.loc[accuracy_df["accuracy"].idxmax()]
    classe_pior = accuracy_df.loc[accuracy_df["accuracy"].idxmin()]

    col1, col2 = st.columns(2)
    col1.metric(t("best_performance"), classe_melhor["class_name"], f'{classe_melhor["accuracy"]}%')
    col2.metric(t("worst_performance"), classe_pior["class_name"], f'{classe_pior["accuracy"]}%')

    st.dataframe(
        accuracy_df.rename(columns={
            "class_id": t("column_class"),
            "class_name": t("column_name"),
            "accuracy": t("column_accuracy")
        }),
        width="stretch",
        hide_index=True
    )

    st.subheader(t("confusion_matrix"))

    confusion_df = gerar_dataframe_matriz(
        y_test,
        y_pred_test,
        st.session_state.language
    )

    confusion_fig = gerar_grafico_matriz(
        confusion_df,
        st.session_state.language
    )

    col_conf_graph, _, col_conf_table = st.columns([1, 0.1, 1.25])

    with col_conf_graph:
        st.pyplot(confusion_fig, width="stretch")

    with col_conf_table:
        st.info(t("matrix_info"))
        st.dataframe(confusion_df, width="stretch")

# ABA 2
with tab2:
    st.subheader(t("test_dataset"))
    st.write(t("test_description"))

    col_filter1, col_filter2, col_button = st.columns([1, 1, 0.35])

    with col_filter1:
        selected_class_id = st.selectbox(
            t("filter_by_class"),
            ["all"] + list(range(10)),
            format_func=lambda option: (
                t("class_all")
                if option == "all"
                else get_class_name(option, st.session_state.language)
            )
        )

    with col_filter2:
        selected_result = st.selectbox(
            t("filter_by_result"),
            ["all", "correct", "wrong"],
            format_func=lambda option: t(f"result_{option}")
        )

    with col_button:
        st.write("")
        st.write("")
        sortear = st.button(t("button_sort"), key="random_filtered", width="stretch")

    indices_por_classe = filtrar_indices_por_classe(test_data, selected_class_id)

    acertos, erros = contar_resultados(
        indices_por_classe,
        test_data,
        y_pred_series
    )

    quantidade_disponivel = {
        "all": len(indices_por_classe),
        "correct": acertos,
        "wrong": erros
    }[selected_result]

    st.caption(f"{t('images_available')} {quantidade_disponivel}")

    if "visual_sample_index" not in st.session_state:
        st.session_state.visual_sample_index = None

    if "historico_indices" not in st.session_state:
        st.session_state.historico_indices = set()

    if sortear:
        indices_para_sorteio = indices_por_classe.copy()
        np.random.shuffle(indices_para_sorteio)

        if selected_result == "all":
            novo_indice = sortear_indice_sem_repetir(
                indices_para_sorteio,
                st.session_state.historico_indices
            )
        else:
            novo_indice = encontrar_resultado(
                indices_para_sorteio,
                test_data,
                y_pred_series,
                selected_result
            )

        st.session_state.visual_sample_index = novo_indice

        if novo_indice is not None:
            st.session_state.historico_indices.add(novo_indice)

        if len(st.session_state.historico_indices) > 80:
            st.session_state.historico_indices.clear()

    sample_index = st.session_state.visual_sample_index

    if len(indices_por_classe) == 0:
        st.warning(t("no_images_found"))

    elif sample_index is None or sample_index not in indices_por_classe:
        st.info(t("click_to_sort"))

    else:
        sample = test_data.loc[sample_index]
        exibir_previsao_dataset(sample, model, st.session_state.language)

    st.divider()

    st.subheader(t("test_upload"))

    with st.expander(t("instructions_title")):
        st.markdown(t("instructions_text"))

    uploaded_file = st.file_uploader(
        t("image_formats"),
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        original_image, processed_image, uploaded_pixels = preprocessar_imagem_upload(uploaded_file)

        exibir_previsao_upload(
            original_image,
            processed_image,
            uploaded_pixels,
            model,
            st.session_state.language
        )