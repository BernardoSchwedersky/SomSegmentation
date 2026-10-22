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

import numpy as np

import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


# Classes da banda SCL (Scene Classification) consideradas inválidas:
# 3=sombra de nuvem, 8=nuvem média, 9=nuvem densa, 10=cirrus
CLASSES_NUVEM_SCL = [3, 8, 9, 10]


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
def ler_asset(item, asset, bbox, resolucao=10, resampling=Resampling.bilinear):
    with rasterio.open(item.assets[asset].href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        janela = from_bounds(*bounds, transform=src.transform)
        altura = int(round((bounds[3] - bounds[1]) / resolucao))
        largura = int(round((bounds[2] - bounds[0]) / resolucao))
        return src.read(1, window=janela, out_shape=(altura, largura),
                        resampling=resampling).astype(float)

# Lê as bandas e converte os números digitais para reflectância (0 a 1). A partir
# da baseline de processamento 04.00 (2022) os valores têm um deslocamento de +1000
def ler_bandas(item, bbox, bandas, resolucao=10):
    img = np.stack([ler_asset(item, banda, bbox, resolucao) for banda in bandas], axis=2)
    offset = 1000 if float(item.properties.get("s2:processing_baseline", "0")) >= 4 else 0
    return np.clip((img - offset) / 10000, 0, 1)

# Máscara booleana dos pixels com nuvem ou sombra de nuvem, a partir da banda SCL
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
