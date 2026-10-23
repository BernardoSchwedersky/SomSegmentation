# -*- coding: utf-8 -*-
"""
Funções para buscar e ler imagens Sentinel-2 do Microsoft Planetary Computer
(coleção "sentinel-2-l2a", reflectância de superfície), usadas nos exemplos
exemploSentinel.py e exemploSentinelEnchente.py.

Dependências extras (além das usadas pelo somSeg):
    pip install pystac-client planetary-computer rasterio

O Planetary Computer não exige chave de API para leitura: o pacote
planetary_computer apenas assina as URLs dos arquivos (tokens SAS temporários).
"""

import time
from datetime import datetime, timedelta

import numpy as np

import planetary_computer
import pystac_client
import rasterio
import rasterio.errors
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


# Classes da banda SCL (Scene Classification) consideradas inválidas:
# 0=sem dado (fora da cena), 1=pixel saturado ou defeituoso, 3=sombra de nuvem,
# 8=nuvem média, 9=nuvem densa, 10=cirrus
CLASSES_NUVEM_SCL = [0, 1, 3, 8, 9, 10]


# Retorna o retângulo [lon_min, lat_min, lon_max, lat_max] em torno de um ponto
def criar_bbox(centro_lon, centro_lat, meia_largura_graus):
    return [centro_lon - meia_largura_graus, centro_lat - meia_largura_graus,
            centro_lon + meia_largura_graus, centro_lat + meia_largura_graus]

# Busca as cenas que cobrem o bbox no período e retorna a com menos nuvens
def buscar_cena(bbox, periodo, nuvens_max):
    catalogo = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    busca = catalogo.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=periodo,
        query={"eo:cloud_cover": {"lt": nuvens_max}},
    )
    itens = list(busca.items())
    if len(itens) == 0:
        raise RuntimeError("Nenhuma cena encontrada em " + periodo + ": aumente o período ou a cobertura de nuvens máxima")
    return min(itens, key=lambda item: item.properties["eo:cloud_cover"])

# Lê apenas a região de interesse de um asset (os arquivos são COGs, então só os
# blocos necessários são baixados), reamostrando para a grade de 10 m
# Falhas momentâneas de rede (ex.: HTTP 403 ou conexão interrompida) são repetidas
def ler_asset(item, asset, bbox, resolucao=10, resampling=Resampling.bilinear, tentativas=4):
    for tentativa in range(tentativas):
        try:
            return _ler_asset(item, asset, bbox, resolucao, resampling)
        except rasterio.errors.RasterioIOError:
            if tentativa == tentativas - 1:
                raise
            time.sleep(5 * 2 ** tentativa)

def _ler_asset(item, asset, bbox, resolucao, resampling):
    with rasterio.open(item.assets[asset].href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        janela = from_bounds(*bounds, transform=src.transform)
        altura = int(round((bounds[3] - bounds[1]) / resolucao))
        largura = int(round((bounds[2] - bounds[0]) / resolucao))
        # Se a região passa da borda da cena, a leitura precisa ser boundless: os pixels
        # de fora ficam com 0 (sem dado). Sem isso, a janela é cortada na borda e a parte
        # de dentro é esticada para o tamanho pedido
        dentro = (janela.col_off >= 0 and janela.row_off >= 0 and janela.col_off + janela.width <= src.width
                  and janela.row_off + janela.height <= src.height)
        return src.read(1, window=janela, out_shape=(altura, largura), resampling=resampling,
                        boundless=not dentro, fill_value=0).astype(float)

# Lê as bandas e converte os números digitais para reflectância (0 a 1). A partir
# da baseline de processamento 04.00 (2022) os valores têm um deslocamento de +1000
def ler_bandas(item, bbox, bandas, resolucao=10):
    img = np.stack([ler_asset(item, banda, bbox, resolucao) for banda in bandas], axis=2)
    offset = 1000 if float(item.properties.get("s2:processing_baseline", "0")) >= 4 else 0
    return np.clip((img - offset) / 10000, 0, 1)

# Máscara booleana dos pixels inválidos (nuvem, sombra ou sem dado), a partir da banda SCL
def ler_mascara_nuvens(item, bbox, resolucao=10):
    scl = ler_asset(item, "SCL", bbox, resolucao, resampling=Resampling.nearest)
    return np.isin(scl, CLASSES_NUVEM_SCL)

# Índice normalizado (a - b) / (a + b), ex.: NDVI = (B08 - B04) / (B08 + B04)
def indice_normalizado(a, b):
    return (a - b) / np.maximum(a + b, 1e-6)

# Os pesos do SOM são inicializados entre 0 e 255, então cada canal é
# reescalado para essa faixa (cortando os percentis extremos)
def normalizar_0_255(img, p_min=2, p_max=98):
    saida = np.zeros(img.shape)
    for i in range(0, img.shape[2]):
        lo, hi = np.percentile(img[:, :, i], [p_min, p_max])
        saida[:, :, i] = np.clip((img[:, :, i] - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return saida

# Fração dos pixels inválidos (nuvem, sombra ou sem dado) de uma cena dentro do
# bbox, lida em baixa resolução para ser rápida
def fracao_invalida(item, bbox, resolucao=60):
    return ler_mascara_nuvens(item, bbox, resolucao).mean()

# Busca todas as cenas que cobrem o bbox no período. Cenas reprocessadas aparecem
# mais de uma vez: mantém só a versão mais recente de cada aquisição e tile
def buscar_cenas(bbox, inicio, fim, nuvens_max=100):
    catalogo = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    busca = catalogo.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=inicio.strftime("%Y-%m-%d") + "/" + fim.strftime("%Y-%m-%d"),
        query={"eo:cloud_cover": {"lt": nuvens_max}},
    )
    cenas = {}
    for item in busca.items():
        chave = (item.datetime.date(), item.properties["s2:mgrs_tile"])
        if chave not in cenas or item.id > cenas[chave].id:
            cenas[chave] = item
    return list(cenas.values())

# Escolhe automaticamente as cenas de depois e de antes de um evento, pela fração
# de pixels inválidos dentro do bbox (e não pela nuvem da cena inteira).
# - Depois: entre dias_depois[0] e dias_depois[1] dias da data do evento,
#   penalizando a distância da data em penalidade_dia_depois por dia
# - Antes: ver selecionar_cena_antes
# Retorna (item_antes, item_depois, fração inválida antes, fração inválida depois)
def selecionar_cenas(bbox, data_evento, dias_depois=(-1, 5), penalidade_dia_depois=0.03, **opcoes_antes):
    data = datetime.strptime(data_evento, "%Y-%m-%d")
    candidatas = buscar_cenas(bbox, data + timedelta(days=dias_depois[0]), data + timedelta(days=dias_depois[1]))
    if not candidatas:
        raise RuntimeError("Nenhuma cena perto de " + data_evento)
    notas = []
    for item in candidatas:
        invalida = fracao_invalida(item, bbox)
        dias = abs((item.datetime.replace(tzinfo=None) - data).days)
        notas.append((invalida + penalidade_dia_depois * dias, invalida, item))
    nota, invalida_depois, item_depois = min(notas, key=lambda n: n[0])
    item_antes, invalida_antes = selecionar_cena_antes(bbox, data_evento, item_depois, **opcoes_antes)
    return item_antes, item_depois, invalida_antes, invalida_depois

# Escolhe a cena de antes do evento: a mais próxima do evento (anterior à data do
# evento, até dias_max dias antes) com no máximo invalida_max de pixels inválidos
# dentro do bbox. Se nenhuma atende, usa a de menor fração inválida. A cena é do
# mesmo tile da cena de depois, para os pixels ficarem alinhados.
# Retorna (item_antes, fração inválida)
def selecionar_cena_antes(bbox, data_evento, item_depois, dias_max=180, invalida_max=0.10, nuvens_max_cena=80):
    data = datetime.strptime(data_evento, "%Y-%m-%d")
    tile = item_depois.properties["s2:mgrs_tile"]
    candidatas = [item for item in buscar_cenas(bbox, data - timedelta(days=dias_max), data - timedelta(days=1),
                                                nuvens_max_cena)
                  if item.properties["s2:mgrs_tile"] == tile and item.datetime < item_depois.datetime]
    if not candidatas:
        raise RuntimeError("Nenhuma cena anterior ao evento no tile " + tile)
    candidatas.sort(key=lambda item: item.datetime, reverse=True)  # Da mais recente para a mais antiga
    melhor = None
    for item in candidatas:
        invalida = fracao_invalida(item, bbox)
        if invalida <= invalida_max:
            return item, invalida
        if melhor is None or invalida < melhor[1]:
            melhor = (item, invalida)
    return melhor

# Busca a cena de uma data específica no mesmo tile de item_depois (ex.: para fixar
# a cena de antes de um evento). Retorna (item, fração inválida dentro do bbox)
def buscar_cena_data(bbox, data, item_depois):
    dia = datetime.strptime(data, "%Y-%m-%d")
    tile = item_depois.properties["s2:mgrs_tile"]
    candidatas = [item for item in buscar_cenas(bbox, dia, dia + timedelta(days=1))
                  if item.properties["s2:mgrs_tile"] == tile and item.datetime.date() == dia.date()]
    if not candidatas:
        raise RuntimeError("Nenhuma cena em %s no tile %s" % (data, tile))
    return candidatas[0], fracao_invalida(candidatas[0], bbox)

# Busca uma cena pelo identificador (ex.: para repetir um processamento com as mesmas cenas)
def buscar_item(id_cena):
    catalogo = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    item = next(catalogo.search(collections=["sentinel-2-l2a"], ids=[id_cena]).items(), None)
    if item is None:
        raise RuntimeError("Cena não encontrada: " + id_cena)
    return item
