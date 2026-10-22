# -*- coding: utf-8 -*-
"""
Funções do mapeamento de enchente com SOM bi-temporal, usadas nos exemplos
exemploSentinelEnchente.py (uma área), exemploSentinelGrade.py (grade de áreas),
exemploComparacaoNDWI.py e processarEventos.py (comparação com o NDWI direto).

As duas datas (antes e depois do evento) são empilhadas em uma única imagem
multicanal, então cada pixel descreve uma trajetória de mudança e cada unidade
do SOM agrupa pixels que mudaram da mesma forma. Cada unidade é depois
classificada pelo índice de água médio dos seus pixels nas duas datas em uma de
três classes: não alagamento, água permanente ou alagamento.

O SOM usa o MNDWI = (B03 - B11) / (B03 + B11) como índice de água. O NDWI
(B03 - B08) / (B03 + B08), usado como baseline na comparação, falha em água
barrenta: o sedimento aumenta a reflectância no infravermelho próximo e o NDWI
fica negativo, enquanto o infravermelho de ondas curtas (B11) continua sendo
absorvido pela água.
"""

import os

import numpy as np

import somSeg
import sentinelUtils


# Bandas usadas: B02=azul, B03=verde, B04=vermelho, B08=infravermelho próximo,
# B11=infravermelho de ondas curtas
BANDAS = ["B02", "B03", "B04", "B08", "B11"]

# Classes atribuídas às unidades do SOM e suas cores nos mapas
CLASSES = ["Não alagamento", "Água permanente", "Alagamento"]
CORES_CLASSES = ["#d9d9d9", "#08306b", "#3fa9f5"]
CLASSES_ENCHENTE = [2]  # Classes somadas na área atingida pela enchente

# Índice usado pelo SOM para identificar a água: "MNDWI" ou "NDWI"
INDICE_AGUA = "MNDWI"
INDICE_BASELINE = "NDWI"  # Índice do método direto usado como baseline

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

# Índice de água de uma data (> 0 indica água): MNDWI usa B03 e B11, NDWI usa B03 e B08
def indice_agua(refl, indice=INDICE_AGUA):
    b03 = refl[:, :, BANDAS.index("B03")]
    outra = refl[:, :, BANDAS.index("B11" if indice == "MNDWI" else "B08")]
    return sentinelUtils.indice_normalizado(b03, outra)

# Imagem de entrada bi-temporal: bandas + índice de água de antes e de depois, em 0-255
def montar_entrada(refl_antes, refl_depois, indice=INDICE_AGUA):
    img = np.concatenate((refl_antes, indice_agua(refl_antes, indice)[:, :, None],
                          refl_depois, indice_agua(refl_depois, indice)[:, :, None]), axis=2)
    nomes_canais = [b + " antes" for b in BANDAS + [indice]] + [b + " depois" for b in BANDAS + [indice]]
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

# Classe de cada pixel a partir da água nas duas datas: água antes e depois é
# água permanente, água só depois é alagamento e o resto é não alagamento
def classe_por_agua(agua_antes, agua_depois):
    return np.where(agua_depois, np.where(agua_antes, 1, 2), 0)

# Atribui uma classe a cada unidade a partir do índice de água médio dos seus
# pixels nas duas datas. Retorna a classe de cada unidade e uma tabela com
# (unidade, pixels, índice antes, índice depois, classe)
def classificar_unidades(result, nuvens, refl_antes, refl_depois, som_size, limiar_agua=0, indice=INDICE_AGUA):
    agua_antes = indice_agua(refl_antes, indice)
    agua_depois = indice_agua(refl_depois, indice)
    classe_unidade = np.zeros(som_size, dtype=int)
    tabela = []
    for u in range(som_size):
        m = (result == u) & ~nuvens
        if not np.any(m):
            continue
        wa, wd = agua_antes[m].mean(), agua_depois[m].mean()
        classe_unidade[u] = classe_por_agua(wa > limiar_agua, wd > limiar_agua)
        tabela.append((u, m.sum(), wa, wd, classe_unidade[u]))
    return classe_unidade, tabela

def imprimir_tabela(tabela, indice=INDICE_AGUA):
    print("Unidade | pixels | %6s antes -> depois | classe" % indice)
    for u, n, wa, wd, c in tabela:
        print("%7d | %6d | %5.2f -> %5.2f           | %s" % (u, n, wa, wd, CLASSES[c]))

# Área (km²) de cada classe no mapa de mudança, sem os pixels com nuvem
def areas_por_classe(mapa_mudanca, nuvens):
    return np.array([((mapa_mudanca == c) & ~nuvens).sum() * AREA_PIXEL_KM2 for c in range(len(CLASSES))])

# Método direto usado como baseline: limiar do índice de água (NDWI) em cada
# pixel das duas datas, com as mesmas classes do SOM
def mapa_indice(refl_antes, refl_depois, limiar_agua=0, indice=INDICE_BASELINE):
    return classe_por_agua(indice_agua(refl_antes, indice) > limiar_agua,
                           indice_agua(refl_depois, indice) > limiar_agua)


# Funções da grade de áreas (exemploSentinelGrade.py, exemploComparacaoNDWI.py e processarEventos.py)

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
