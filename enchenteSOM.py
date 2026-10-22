# -*- coding: utf-8 -*-
"""
Funções do mapeamento de enchente com SOM bi-temporal, usadas nos exemplos
exemploSentinelEnchente.py (uma área) e exemploSentinelGrade.py (grade de áreas).

As duas datas (antes e depois do evento) são empilhadas em uma única imagem
multicanal, então cada pixel descreve uma trajetória de mudança e cada unidade
do SOM agrupa pixels que mudaram da mesma forma. As unidades são depois
interpretadas a partir dos índices espectrais médios dos seus pixels.
"""

import os

import numpy as np

import somSeg
import sentinelUtils


# Bandas usadas: B02=azul, B03=verde, B04=vermelho, B08=infravermelho próximo,
# B11=infravermelho de ondas curtas
BANDAS = ["B02", "B03", "B04", "B08", "B11"]

# Classes de mudança atribuídas às unidades do SOM e suas cores nos mapas
CLASSES = ["Sem mudança", "Água permanente", "Área inundada",
           "Vegetação afetada (possível alagamento)", "Água que recuou"]
CORES_CLASSES = ["#d9d9d9", "#08306b", "#3fa9f5", "#d95f0e", "#74c476"]
CLASSES_ENCHENTE = [2, 3]  # Classes somadas na área atingida pela enchente

AREA_PIXEL_KM2 = 10 * 10 / 1e6


# Lê as bandas das duas cenas e a máscara de nuvens (nuvem ou sombra em qualquer
# uma das datas). Se arquivo_cache for informado, salva/lê os dados dele
def ler_dados(item_antes, item_depois, bbox, arquivo_cache=None):
    if arquivo_cache is not None and os.path.exists(arquivo_cache):
        dados = np.load(arquivo_cache)
        return dados["refl_antes"], dados["refl_depois"], dados["nuvens"]
    refl_antes = sentinelUtils.ler_bandas(item_antes, bbox, BANDAS)
    refl_depois = sentinelUtils.ler_bandas(item_depois, bbox, BANDAS)
    if refl_antes.shape != refl_depois.shape:
        raise RuntimeError("As cenas estão em grades diferentes: escolha cenas do mesmo tile")
    nuvens = sentinelUtils.ler_mascara_nuvens(item_antes, bbox) | sentinelUtils.ler_mascara_nuvens(item_depois, bbox)
    if arquivo_cache is not None:
        np.savez_compressed(arquivo_cache, refl_antes=refl_antes, refl_depois=refl_depois, nuvens=nuvens)
    return refl_antes, refl_depois, nuvens

# Índices espectrais de uma data: NDVI (vegetação) e NDWI (água, > 0 indica água)
def indices(refl):
    b03, b04, b08 = [refl[:, :, BANDAS.index(b)] for b in ["B03", "B04", "B08"]]
    return sentinelUtils.indice_normalizado(b08, b04), sentinelUtils.indice_normalizado(b03, b08)

# Imagem de entrada bi-temporal: bandas + NDWI de antes e de depois, em 0-255
def montar_entrada(refl_antes, refl_depois):
    ndwi_antes = indices(refl_antes)[1]
    ndwi_depois = indices(refl_depois)[1]
    img = np.concatenate((refl_antes, ndwi_antes[:, :, None],
                          refl_depois, ndwi_depois[:, :, None]), axis=2)
    nomes_canais = [b + " antes" for b in BANDAS + ["NDWI"]] + [b + " depois" for b in BANDAS + ["NDWI"]]
    return sentinelUtils.normalizar_0_255(img), nomes_canais

# Composição em cor verdadeira para visualização (mesmo realce em todas as datas)
def cor_verdadeira(refl):
    return np.clip(refl[:, :, [BANDAS.index("B04"), BANDAS.index("B03"), BANDAS.index("B02")]] / 0.15, 0, 1)

# Treina o SOM com uma porcentagem dos pixels sem nuvem e segmenta a imagem toda
def treinar_e_segmentar(ROI_sample_conv, nuvens, patch_size=1, som_size=12, epochs=3,
                        dataset_percentage=0.2, seed=0):
    numero_canais = ROI_sample_conv.shape[2]
    all_data, imgInputSOM = somSeg.prepare_dataset_multichanel(ROI_sample_conv, patch_size)
    validos = np.flatnonzero(~nuvens.ravel())
    selecionados = np.random.RandomState(seed).choice(validos, int(len(validos) * dataset_percentage), replace=False)
    train_data = all_data[selecionados]
    del all_data

    somSeg.rand.seed(seed)  # Mesma inicialização do SOM em todas as execuções
    SOM = somSeg.create_map(som_size, 1, (2*patch_size+1)*(2*patch_size+1)*numero_canais)
    SOM = somSeg.train_SOM(SOM, train_data, epochs=epochs)

    # Cada pixel recebe o índice da unidade vencedora (BMU)
    result = somSeg.evaluate_SOM(SOM, ROI_sample_conv, patch_size)
    return SOM, result

# Atribui uma classe de mudança a cada unidade a partir dos índices médios dos
# seus pixels nas duas datas. Retorna a classe de cada unidade e uma tabela
# com (unidade, pixels, NDWI antes, NDWI depois, NDVI antes, NDVI depois, classe)
def classificar_unidades(result, nuvens, refl_antes, refl_depois, som_size,
                         limiar_agua=0, limiar_perda_ndvi=0.15, ndvi_min_vegetacao=0.5):
    ndvi_antes, ndwi_antes = indices(refl_antes)
    ndvi_depois, ndwi_depois = indices(refl_depois)
    classe_unidade = np.zeros(som_size, dtype=int)
    tabela = []
    for u in range(som_size):
        m = (result == u) & ~nuvens
        if not np.any(m):
            continue
        wa, wd = ndwi_antes[m].mean(), ndwi_depois[m].mean()
        va, vd = ndvi_antes[m].mean(), ndvi_depois[m].mean()
        if wa > limiar_agua and wd > limiar_agua:
            classe_unidade[u] = 1
        elif wd > limiar_agua:
            classe_unidade[u] = 2
        elif va > ndvi_min_vegetacao and va - vd > limiar_perda_ndvi:
            classe_unidade[u] = 3
        elif wa > limiar_agua:
            classe_unidade[u] = 4
        tabela.append((u, m.sum(), wa, wd, va, vd, classe_unidade[u]))
    return classe_unidade, tabela

def imprimir_tabela(tabela):
    print("Unidade | pixels | NDWI antes -> depois | NDVI antes -> depois | classe")
    for u, n, wa, wd, va, vd, c in tabela:
        print("%7d | %6d | %5.2f -> %5.2f        | %5.2f -> %5.2f        | %s"
              % (u, n, wa, wd, va, vd, CLASSES[c]))

# Área (km²) de cada classe no mapa de mudança, sem os pixels com nuvem
def areas_por_classe(mapa_mudanca, nuvens):
    return np.array([((mapa_mudanca == c) & ~nuvens).sum() * AREA_PIXEL_KM2 for c in range(len(CLASSES))])

# Método direto para comparação: limiar de NDWI em cada data, com os mesmos
# códigos de classe do SOM (1 água permanente, 2 área inundada, 4 água que recuou).
# Não tem equivalente à classe 3 (vegetação afetada)
def mapa_ndwi(refl_antes, refl_depois, limiar_agua=0):
    agua_antes = indices(refl_antes)[1] > limiar_agua
    agua_depois = indices(refl_depois)[1] > limiar_agua
    mapa = np.zeros(agua_antes.shape, dtype=int)
    mapa[agua_antes & agua_depois] = 1
    mapa[~agua_antes & agua_depois] = 2
    mapa[agua_antes & ~agua_depois] = 4
    return mapa


# Funções da grade de áreas (exemploSentinelGrade.py e exemploComparacaoNDWI.py)

# Retângulo da área (i, j) de uma grade n_linhas x n_colunas centrada em
# (centro_lon, centro_lat). A linha 0 é a mais ao norte
def bbox_area_grade(i, j, centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas):
    lon = centro_lon + (j - (n_colunas - 1) / 2) * 2 * meia_largura_graus
    lat = centro_lat + ((n_linhas - 1) / 2 - i) * 2 * meia_largura_graus
    return sentinelUtils.criar_bbox(lon, lat, meia_largura_graus)

# Retângulo que envolve a grade inteira
def bbox_grade(centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas):
    so = bbox_area_grade(n_linhas - 1, 0, centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
    ne = bbox_area_grade(0, n_colunas - 1, centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
    return [so[0], so[1], ne[2], ne[3]]

# Nome da área pela posição na grade (ex.: "Norte", "Sudeste", "Centro")
def nome_area_grade(i, j, n_linhas, n_colunas):
    ns = "Norte" if i < (n_linhas - 1) / 2 else "Sul" if i > (n_linhas - 1) / 2 else ""
    lo = "Leste" if j > (n_colunas - 1) / 2 else "Oeste" if j < (n_colunas - 1) / 2 else ""
    combinados = {("Norte", "Leste"): "Nordeste", ("Norte", "Oeste"): "Noroeste",
                  ("Sul", "Leste"): "Sudeste", ("Sul", "Oeste"): "Sudoeste"}
    nome = combinados.get((ns, lo), ns or lo or "Centro")
    # Em grades maiores que 3x3 vários nomes se repetem: inclui a posição
    if max(n_linhas, n_colunas) > 3:
        nome += " (%d,%d)" % (i, j)
    return nome

# Máscara de nuvens de uma área, lida do cache se ele existir. Serve para
# conferir as nuvens antes de baixar as bandas
def ler_nuvens(item_antes, item_depois, bbox, arquivo_cache=None):
    if arquivo_cache is not None and os.path.exists(arquivo_cache):
        return np.load(arquivo_cache)["nuvens"]
    return sentinelUtils.ler_mascara_nuvens(item_antes, bbox) | sentinelUtils.ler_mascara_nuvens(item_depois, bbox)
