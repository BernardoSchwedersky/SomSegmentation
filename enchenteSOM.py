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

# Número padrão de unidades do SOM (testado de 4 a 128 no Sen1Floods11, variacaoUnidadesSOM.py)
SOM_SIZE = 36


# Lê o cache se ele existir e for das mesmas cenas (o cache guarda os ids delas)
def _ler_cache(arquivo_cache, item_antes, item_depois):
    if arquivo_cache is None or not os.path.exists(arquivo_cache):
        return None
    dados = np.load(arquivo_cache)
    if "cenas" not in dados or list(dados["cenas"]) != [item_antes.id, item_depois.id]:
        return None
    return dados

# Lê as bandas das duas cenas e a máscara de nuvens (nuvem ou sombra em qualquer
# uma das datas). Se arquivo_cache for informado, salva/lê os dados dele
def ler_dados(item_antes, item_depois, bbox, arquivo_cache=None):
    dados = _ler_cache(arquivo_cache, item_antes, item_depois)
    if dados is not None and "refl_antes" in dados:
        return dados["refl_antes"], dados["refl_depois"], dados["nuvens"]
    refl_antes = sentinelUtils.ler_bandas(item_antes, bbox, BANDAS)
    refl_depois = sentinelUtils.ler_bandas(item_depois, bbox, BANDAS)
    if refl_antes.shape != refl_depois.shape:
        raise RuntimeError("As cenas estão em grades diferentes: escolha cenas do mesmo tile")
    nuvens = sentinelUtils.ler_mascara_nuvens(item_antes, bbox) | sentinelUtils.ler_mascara_nuvens(item_depois, bbox)
    if arquivo_cache is not None:
        np.savez_compressed(arquivo_cache, refl_antes=refl_antes, refl_depois=refl_depois, nuvens=nuvens,
                            cenas=[item_antes.id, item_depois.id])
    return refl_antes, refl_depois, nuvens

# Índice de água de uma data (> 0 indica água): MNDWI usa B03 e B11, NDWI usa B03 e B08
def indice_agua(refl, indice=INDICE_AGUA):
    b03 = refl[:, :, BANDAS.index("B03")]
    outra = refl[:, :, BANDAS.index("B11" if indice == "MNDWI" else "B08")]
    return sentinelUtils.indice_normalizado(b03, outra)

# Imagem de entrada bi-temporal: bandas + índice de água de antes e de depois, em 0-255.
# extras: canais adicionais (antes, depois), ex.: features de filtros convolucionais
# (featuresCNN.features_pca), acrescentados a cada data
def montar_entrada(refl_antes, refl_depois, indice=INDICE_AGUA, extras=None):
    partes = [refl_antes, indice_agua(refl_antes, indice)[:, :, None]]
    partes_depois = [refl_depois, indice_agua(refl_depois, indice)[:, :, None]]
    nomes = BANDAS + [indice]
    if extras is not None:
        partes.append(extras[0])
        partes_depois.append(extras[1])
        nomes = nomes + ["extra %d" % (i + 1) for i in range(extras[0].shape[2])]
    img = np.concatenate(partes + partes_depois, axis=2)
    nomes_canais = [n + " antes" for n in nomes] + [n + " depois" for n in nomes]
    return sentinelUtils.normalizar_0_255(img), nomes_canais

# Composição em cor verdadeira para visualização (mesmo realce em todas as datas)
# A reflectância RGB_MAX vira branco; em regiões claras (deserto, Sahel) use um valor maior
RGB_MAX = 0.15

def cor_verdadeira(refl, rgb_max=None):
    return np.clip(refl[:, :, [BANDAS.index("B04"), BANDAS.index("B03"), BANDAS.index("B02")]] / (rgb_max or RGB_MAX), 0, 1)

# Treina o SOM com uma porcentagem dos pixels sem nuvem (padrão 2%, sorteada) e
# segmenta a imagem toda. Usa a versão amostrada e vetorizada abaixo: o resultado
# é igual ao de montar as janelas de todos os pixels, e 2% dá o mesmo resultado
# que 1% a 100% nos testes do Sen1Floods11 (experimentoAmostragem.py)
def treinar_e_segmentar(ROI_sample_conv, nuvens, patch_size=1, som_size=SOM_SIZE, epochs=3,
                        dataset_percentage=0.02, seed=0):
    SOM, result, _ = treinar_e_segmentar_amostrado(ROI_sample_conv, nuvens, dataset_percentage, patch_size, som_size,
                                                   epochs, seed)
    return SOM, result

# Classe de cada pixel a partir da água nas duas datas: água antes e depois é
# água permanente, água só depois é alagamento e o resto é não alagamento
def classe_por_agua(agua_antes, agua_depois):
    return np.where(agua_depois, np.where(agua_antes, 1, 2), 0)

# Janelas (2*patch_size+1)x(2*patch_size+1) de todos os canais em volta dos pixels
# indicados (índices na imagem achatada), no mesmo formato de
# somSeg.prepare_dataset_multichanel (canal a canal, cada janela por linhas, borda
# repetida), mas sem montar as janelas de todos os pixels
def extrair_janelas(img, patch_size, indices):
    k = 2 * patch_size + 1
    pad = np.pad(img, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), mode="edge")
    janelas = np.lib.stride_tricks.sliding_window_view(pad, (k, k), axis=(0, 1))  # (H, W, C, k, k)
    linhas, colunas = np.unravel_index(indices, img.shape[:2])
    return janelas[linhas, colunas].reshape(len(indices), -1)

# Unidade vencedora (BMU) de cada pixel, calculada em blocos de forma vetorizada
# (mesmo resultado de somSeg.evaluate_SOM, que percorre pixel a pixel)
def segmentar(SOM, img, patch_size, bloco=65536):
    pesos = SOM["weights"][:, 0, :]
    n = img.shape[0] * img.shape[1]
    result = np.empty(n, dtype=int)
    for inicio in range(0, n, bloco):
        x = extrair_janelas(img, patch_size, np.arange(inicio, min(inicio + bloco, n)))
        dist = (x ** 2).sum(1)[:, None] - 2 * x @ pesos.T + (pesos ** 2).sum(1)[None, :]
        result[inicio:inicio + len(x)] = np.argmin(dist, axis=1)
    return result.reshape(img.shape[:2])

# Variação com amostragem: treina o SOM só com uma porcentagem dos pixels sem
# nuvem, sorteada aleatoriamente, montando as janelas apenas desses pixels, e
# segmenta a imagem toda de forma vetorizada. Retorna também os tempos (s) do
# treino (sorteio + janelas + treinamento) e da segmentação
def treinar_e_segmentar_amostrado(ROI_sample_conv, nuvens, porcentagem=0.02, patch_size=1, som_size=SOM_SIZE, epochs=3,
                                  seed=0):
    import time
    t0 = time.perf_counter()
    validos = np.flatnonzero(~nuvens.ravel())
    n = max(int(round(len(validos) * porcentagem)), som_size)
    selecionados = np.random.RandomState(seed).choice(validos, min(n, len(validos)), replace=False)
    train_data = extrair_janelas(ROI_sample_conv, patch_size, selecionados)
    somSeg.rand.seed(seed)  # Mesma inicialização do SOM em todas as execuções
    SOM = somSeg.create_map(som_size, 1, train_data.shape[1])
    SOM = somSeg.train_SOM(SOM, train_data, epochs=epochs)
    t1 = time.perf_counter()
    result = segmentar(SOM, ROI_sample_conv, patch_size)
    t2 = time.perf_counter()
    return SOM, result, {"treino_s": t1 - t0, "segmentacao_s": t2 - t1, "amostras": len(selecionados)}

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
    dados = _ler_cache(arquivo_cache, item_antes, item_depois)
    if dados is not None:
        return dados["nuvens"]
    return sentinelUtils.ler_mascara_nuvens(item_antes, bbox) | sentinelUtils.ler_mascara_nuvens(item_depois, bbox)
