import streamlit as st
import pandas as pd

from PIL import ImageOps
from image_utils import mostrar_imagem, preparar_imagem_para_exibicao, analisar_imagem_upload
from translations import get_text, get_class_name


def calcular_previsao(model, pixels, language):
    """
    Calcula a previsão, a confiança e a tabela de probabilidades de uma imagem.
    """
    pixels_normalizados = pixels / 255.0

    prediction = int(model.predict([pixels_normalizados])[0])
    probabilities = model.predict_proba([pixels_normalizados])[0]
    confidence = probabilities[prediction] * 100

    class_column = get_text(language, "column_class")
    probability_column = get_text(language, "column_probability")

    probability_df = pd.DataFrame({
        class_column: [get_class_name(i, language) for i in range(10)],
        probability_column: probabilities * 100
    })

    probability_df[probability_column] = probability_df[probability_column].round(2)

    return prediction, confidence, probability_df


def exibir_previsao_dataset(sample, model, language):
    """
    Exibe a previsão para uma imagem do dataset de teste.
    """
    true_label = int(sample["label"])
    pixels = sample.drop("label")

    prediction, confidence, probability_df = calcular_previsao(model, pixels, language)

    col_img, _, col_result = st.columns([1, 0.1, 1.25])

    with col_img:
        st.pyplot(mostrar_imagem(pixels), width="content")

    with col_result:
        st.write(f"**{get_text(language, 'true_class')}:** {get_class_name(true_label, language)}")
        st.write(f"**{get_text(language, 'model_prediction')}** {get_class_name(prediction, language)}")
        st.write(f"**{get_text(language, 'prediction_confidence')}** {confidence:.1f}%")

        if prediction == true_label:
            st.success(get_text(language, "correct_prediction"))
        else:
            st.error(get_text(language, "wrong_prediction"))

        class_column = get_text(language, "column_class")
        probability_column = get_text(language, "column_probability")

        st.bar_chart(
            probability_df,
            x=class_column,
            y=probability_column
        )


def exibir_previsao_upload(original_image, processed_image, pixels, model, language):
    """
    Exibe a previsão para uma imagem enviada pelo usuário.
    """
    prediction, confidence, probability_df = calcular_previsao(model, pixels, language)

    mostrar_processada = st.toggle(
        get_text(language, "show_processed_image"),
        value=False
    )

    col_img, _, col_result = st.columns([1, 0.1, 1.25])

    with col_img:
        if mostrar_processada:
            imagem_processada_exibicao = preparar_imagem_para_exibicao(
                processed_image,
                tamanho=550,
                espessura_borda=20
            )

            st.image(
                imagem_processada_exibicao,
                width=550
            )

        else:
            imagem_original_com_borda = ImageOps.expand(
                original_image,
                border=50,
                fill="white"
            )

            st.image(
                imagem_original_com_borda,
                width=550
            )

    with col_result:
        st.write(f"**{get_text(language, 'model_prediction')}** {get_class_name(prediction, language)}")
        st.write(f"**{get_text(language, 'prediction_confidence')}** {confidence:.1f}%")

        st.warning(
            get_text(language, "real_image_warning")
        )

        class_column = get_text(language, "column_class")
        probability_column = get_text(language, "column_probability")

        st.bar_chart(
            probability_df,
            x=class_column,
            y=probability_column
        )

        checklist_imagem(original_image, processed_image, language)


def checklist_imagem(original_image, processed_image, language):
    """
    Exibe um diagnóstico técnico da imagem enviada pelo usuário.
    """
    diagnostico = analisar_imagem_upload(original_image, processed_image, language)

    icones = {
        "ok": "✔️",
        "alerta": "➖",
        "erro": "✖️"
    }

    with st.expander(get_text(language, "technical_checklist")):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            item = diagnostico["proporcao"]
            st.markdown(f"{icones[item['nivel']]} **{get_text(language, 'proportion')}**")
            st.write(item["status"])
            st.caption(item["detalhe"])

        with col2:
            item = diagnostico["contraste"]
            st.markdown(f"{icones[item['nivel']]} **{get_text(language, 'contrast')}**")
            st.write(item["status"])
            st.caption(item["detalhe"])

        with col3:
            item = diagnostico["area_objeto"]
            st.markdown(f"{icones[item['nivel']]} **{get_text(language, 'object_area')}**")
            st.write(item["status"])
            st.caption(item["detalhe"])

        with col4:
            item = diagnostico["centralizacao"]
            st.markdown(f"{icones[item['nivel']]} **{get_text(language, 'centralization')}**")
            st.write(item["status"])
            st.caption(item["detalhe"])

        st.caption(
            get_text(language, "checklist_caption")
        )