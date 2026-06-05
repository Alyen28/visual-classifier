import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from translations import get_text

def mostrar_imagem(pixel_values):
    """
    Converte os 784 pixels de uma imagem em uma matriz 28x28 e retorna uma figura.
    """
    image = np.array(pixel_values).reshape(28, 28)

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.imshow(image, cmap="gray")
    ax.axis("off")

    return fig


def encontrar_mascara(gray_array, fundo_medio, objeto_escuro, threshold=15, tentativa=0, limite_tentativas=8):
    """
    Tenta encontrar uma máscara adequada para separar o objeto do fundo.
    """
    if objeto_escuro:
        mask = gray_array < (fundo_medio - threshold)
    else:
        mask = gray_array > (fundo_medio + threshold)

    area_mascara = np.count_nonzero(mask) / mask.size * 100

    if 5 <= area_mascara <= 80:
        return mask

    if tentativa >= limite_tentativas:
        return None

    if area_mascara < 5:
        novo_threshold = threshold - 5
    else:
        novo_threshold = threshold + 5

    if novo_threshold < 5 or novo_threshold > 80:
        return None

    return encontrar_mascara(gray_array, fundo_medio, objeto_escuro, novo_threshold, tentativa + 1, limite_tentativas)


def preprocessar_imagem_upload(uploaded_file, recortar_objeto=True):
    """
    Processa uma imagem enviada pelo usuário para aproximá-la do formato do Fashion-MNIST.
    """
    original_image = Image.open(uploaded_file).convert("RGB")
    gray_array = np.array(original_image.convert("L"))

    if recortar_objeto:
        bordas = np.concatenate([
            gray_array[0, :],
            gray_array[-1, :],
            gray_array[:, 0],
            gray_array[:, -1]
        ])

        fundo_medio = np.median(bordas)
        objeto_escuro = np.median(gray_array) < fundo_medio

        mascara = encontrar_mascara(gray_array, fundo_medio, objeto_escuro)

        if mascara is not None:
            coords = np.argwhere(mascara)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            altura, largura = gray_array.shape
            padding_y = int((y_max - y_min) * 0.08)
            padding_x = int((x_max - x_min) * 0.08)

            y_min = max(0, y_min - padding_y)
            y_max = min(altura, y_max + padding_y)
            x_min = max(0, x_min - padding_x)
            x_max = min(largura, x_max + padding_x)

            gray_crop = gray_array[y_min:y_max, x_min:x_max]
            mask_crop = mascara[y_min:y_max, x_min:x_max]

            processed_crop = np.zeros_like(gray_crop)
            objeto = 255 - gray_crop if objeto_escuro else gray_crop
            processed_crop[mask_crop] = objeto[mask_crop]
        else:
            processed_crop = 255 - gray_array
    else:
        processed_crop = 255 - gray_array

    processed_pil = Image.fromarray(processed_crop.astype(np.uint8))
    processed_pil = ImageOps.autocontrast(processed_pil)

    largura, altura = processed_pil.size
    tamanho = max(largura, altura)

    canvas = Image.new("L", (tamanho, tamanho), color=0)
    x_offset = (tamanho - largura) // 2
    y_offset = (tamanho - altura) // 2
    canvas.paste(processed_pil, (x_offset, y_offset))

    final_image = canvas.resize((28, 28), Image.Resampling.LANCZOS)

    processed_image = np.array(final_image)
    pixels = processed_image.flatten()

    return original_image, processed_image, pixels


def analisar_imagem_upload(original_image, processed_image, language):
    """
    Analisa a aimgem e verifica se ela está próxima do formato esperado pelo modelo.
    """
    largura, altura = original_image.size
    proporcao = largura / altura

    if 0.95 <= proporcao <= 1.05:
        status_proporcao, nivel_proporcao = get_text(language, "status_adequate"), "ok"
    elif 0.85 <= proporcao <= 1.15:
        status_proporcao, nivel_proporcao = get_text(language, "status_acceptable"), "alerta"
    else:
        status_proporcao, nivel_proporcao = get_text(language, "status_irregular"), "erro"

    gray_array = np.array(original_image.convert("L"))
    contraste = gray_array.std()

    if contraste >= 55:
        status_contraste, nivel_contraste = get_text(language, "status_good"), "ok"
    elif contraste >= 35:
        status_contraste, nivel_contraste = get_text(language, "status_medium"), "alerta"
    else:
        status_contraste, nivel_contraste = get_text(language, "status_low"), "erro"

    mascara = processed_image > 20
    area_objeto = np.count_nonzero(mascara) / processed_image.size * 100

    if 20 <= area_objeto <= 75:
        status_area, nivel_area = get_text(language, "status_adequate"), "ok"
    elif 12 <= area_objeto < 20 or 75 < area_objeto <= 85:
        status_area, nivel_area = get_text(language, "status_acceptable"), "alerta"
    else:
        status_area, nivel_area = get_text(language, "status_inadequate"), "erro"

    coords = np.argwhere(mascara)

    if coords.size > 0:
        centro_y, centro_x = coords.mean(axis=0)
        distancia_centro = np.sqrt((centro_x - 13.5) ** 2 + (centro_y - 13.5) ** 2)

        if distancia_centro <= 3:
            status_centralizacao, nivel_centralizacao = get_text(language, "status_good"), "ok"
        elif distancia_centro <= 5:
            status_centralizacao, nivel_centralizacao = get_text(language, "status_acceptable"), "alerta"
        else:
            status_centralizacao, nivel_centralizacao = get_text(language, "status_displaced"), "erro"

        detalhe_centralizacao = f"{distancia_centro:.1f} {get_text(language, 'center_distance')}"
    else:
        status_centralizacao = get_text(language, "status_undefined")
        nivel_centralizacao = "erro"
        detalhe_centralizacao = get_text(language, "object_not_detected")

    return {
        "proporcao": {
            "status": status_proporcao,
            "nivel": nivel_proporcao,
            "detalhe": f"{largura} x {altura}px"
        },
        "contraste": {
            "status": status_contraste,
            "nivel": nivel_contraste,
            "detalhe": f"{contraste:.1f}"
        },
        "area_objeto": {
            "status": status_area,
            "nivel": nivel_area,
            "detalhe": f"{area_objeto:.1f}%"
        },
        "centralizacao": {
            "status": status_centralizacao,
            "nivel": nivel_centralizacao,
            "detalhe": detalhe_centralizacao
        }
    }
    

def preparar_imagem_para_exibicao(processed_image, tamanho=320, espessura_borda=8):
    """
    Cria uma versão ampliada da imagem processada apenas para exibição.
    """
    imagem_pil = Image.fromarray(processed_image.astype(np.uint8)).convert("L")
    imagem_ampliada = imagem_pil.resize((tamanho, tamanho), Image.Resampling.NEAREST)
    imagem_com_borda = ImageOps.expand(imagem_ampliada, border=espessura_borda, fill="white")

    return imagem_com_borda