# -*- coding: utf-8 -*-
"""
Exemplo de segmentação de imagens de satélite Sentinel-2 com o somSeg.

As imagens são obtidas do Microsoft Planetary Computer (coleção "sentinel-2-l2a",
reflectância de superfície já corrigida atmosfericamente), acessadas via STAC.

Dependências extras (além das usadas pelo somSeg):
    pip install pystac-client planetary-computer rasterio

O Planetary Computer não exige chave de API para leitura: o pacote
planetary_computer apenas assina as URLs dos arquivos (tokens SAS temporários).
"""

import os

import somSeg
import matplotlib.pyplot as plt
import numpy as np

import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


# Parâmetros da busca da imagem
centro_lon, centro_lat = -47.83, -15.80   # Lago Paranoá, Brasília - DF
meia_largura_graus = 0.0125               # ~2,8 km de lado -> ~280x280 pixels de 10 m
periodo = "2024-06-01/2024-08-31"         # Estação seca: menos nuvens
cobertura_nuvens_max = 5                  # Porcentagem máxima de nuvens na cena

# Bandas usadas como canais de entrada do SOM
# B02=azul, B03=verde, B04=vermelho, B08=infravermelho próximo (10 m)
# B11=infravermelho de ondas curtas (20 m, reamostrada para 10 m)
bandas = ["B02", "B03", "B04", "B08", "B11"]

arquivo_cache = "sentinel_cache.npz"      # Evita baixar a imagem novamente


# Busca e leitura da imagem no Planetary Computer
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
        raise RuntimeError("Nenhuma cena encontrada: aumente o período ou a cobertura de nuvens máxima")
    # Escolhe a cena com menos nuvens
    return min(itens, key=lambda item: item.properties["eo:cloud_cover"])

# Lê apenas a região de interesse de cada banda (os arquivos são COGs, então
# só os blocos necessários são baixados). Todas as bandas são reamostradas
# para o tamanho da grade de 10 m.
def ler_bandas(item, bbox, bandas, resolucao=10):
    canais = []
    for banda in bandas:
        with rasterio.open(item.assets[banda].href) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
            janela = from_bounds(*bounds, transform=src.transform)
            if len(canais) == 0:
                altura = int(round((bounds[3] - bounds[1]) / resolucao))
                largura = int(round((bounds[2] - bounds[0]) / resolucao))
            canal = src.read(1, window=janela, out_shape=(altura, largura),
                             resampling=Resampling.bilinear)
            canais.append(canal.astype(float))
    img = np.stack(canais, axis=2)
    # Converte os números digitais para reflectância (0 a 1). A partir da baseline
    # de processamento 04.00 (2022) os valores têm um deslocamento de +1000
    offset = 1000 if float(item.properties.get("s2:processing_baseline", "0")) >= 4 else 0
    return np.clip((img - offset) / 10000, 0, 1)

# Os pesos do SOM são inicializados entre 0 e 255, então cada canal é
# reescalado para essa faixa (cortando os percentis extremos)
def normalizar_0_255(img, p_min=2, p_max=98):
    saida = np.zeros(img.shape)
    for i in range(0, img.shape[2]):
        lo, hi = np.percentile(img[:, :, i], [p_min, p_max])
        saida[:, :, i] = np.clip((img[:, :, i] - lo) / (hi - lo), 0, 1) * 255
    return saida


bbox = [centro_lon - meia_largura_graus, centro_lat - meia_largura_graus,
        centro_lon + meia_largura_graus, centro_lat + meia_largura_graus]

if os.path.exists(arquivo_cache):
    dados = np.load(arquivo_cache)
    reflectancia, nome_cena = dados["reflectancia"], str(dados["nome_cena"])
else:
    item = buscar_cena(bbox, periodo, cobertura_nuvens_max)
    nome_cena = item.id
    print("Cena:", nome_cena, "- nuvens:", item.properties["eo:cloud_cover"], "%")
    reflectancia = ler_bandas(item, bbox, bandas)
    np.savez_compressed(arquivo_cache, reflectancia=reflectancia, nome_cena=nome_cena)

print("Imagem:", reflectancia.shape)


# Criando a img de entrada com os canais desejados (bandas + NDVI)
img_shape = reflectancia.shape
numero_canais = img_shape[2] + 1

b04 = reflectancia[:, :, bandas.index("B04")]
b08 = reflectancia[:, :, bandas.index("B08")]
ndvi = (b08 - b04) / np.maximum(b08 + b04, 1e-6)  # Índice de vegetação, entre -1 e 1

ROI_sample_conv = np.zeros((img_shape[0], img_shape[1], numero_canais))
ROI_sample_conv[:, :, 0:img_shape[2]] = reflectancia
ROI_sample_conv[:, :, img_shape[2]] = ndvi
ROI_sample_conv = normalizar_0_255(ROI_sample_conv)
nomes_canais = bandas + ["NDVI"]

# Composição em cor verdadeira (B04, B03, B02) para visualização
img_rgb = ROI_sample_conv[:, :, [bandas.index("B04"), bandas.index("B03"), bandas.index("B02")]] / 255

# Vizualizando os canais de entrada
plt.figure(figsize=(18, 4))
plt.subplot(1, numero_canais + 1, 1)
plt.imshow(img_rgb)
plt.title(nome_cena[:24], fontsize=8)
for i in range(0, numero_canais):
    plt.subplot(1, numero_canais + 1, i + 2)
    plt.imshow(ROI_sample_conv[:, :, i], cmap="gray")
    plt.title(nomes_canais[i])

# Parâmetros de ajuste do método
patch_size = 1  # Vizinhança de (2*patch_size+1)x(2*patch_size+1) pixels em torno de cada pixel
som_size = 10  # Número de unidades do SOM
epochs = 3  # Quantidade de épocas do treinamento
dataset_percentage = 0.2  # Porcentagem dos pixels usados no treinamento (a avaliação usa todos)

# Criando o dataset para treinar o SOM
train_data, imgInputSOM = somSeg.prepare_dataset_multichanel2(ROI_sample_conv, patch_size, percentage=dataset_percentage)
print("Samples")
print(train_data.shape[0])

# Inicializando o SOM
SOM = somSeg.create_map(som_size, 1, (2*patch_size+1)*(2*patch_size+1)*numero_canais)

# Realizando o treinamento
SOM = somSeg.train_SOM(SOM, train_data, epochs=epochs)

# Segmentação: cada pixel recebe o índice da unidade vencedora (BMU). Como o SOM
# é unidimensional, unidades próximas representam pixels espectralmente parecidos
result = somSeg.evaluate_SOM(SOM, ROI_sample_conv, patch_size)

somSeg.plot_result_multiclass_matplotlib(result, som_size, img_rgb)

somSeg.plot_result_multiclass_plotly(result, som_size)

# Plotando a distância entre as unidades e a quantidade de pixels em cada uma.
# Picos na distância indicam bons limiares para separar classes (ex.: água x terra)
nd_x, nd_y = somSeg.find_ND(SOM['weights'])
hits = somSeg.find_Hits(SOM, ROI_sample_conv, patch_size)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
ax[0].plot(nd_x, marker='o')
ax[0].set_title('Distância entre unidades vizinhas')
ax[1].bar(np.arange(som_size), hits)
ax[1].set_title('Pixels por unidade (hits)')
fig.tight_layout()

# NDVI médio de cada unidade, ajuda a interpretar as classes encontradas
ndvi_unidade = [ndvi[result == u].mean() if np.any(result == u) else np.nan for u in range(som_size)]
print("NDVI médio por unidade:", np.round(ndvi_unidade, 2))

plt.show()
