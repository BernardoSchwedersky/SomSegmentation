# -*- coding: utf-8 -*-
"""
Exemplo de segmentação de imagens de satélite Sentinel-2 com o somSeg.

As imagens são obtidas do Microsoft Planetary Computer (coleção "sentinel-2-l2a",
reflectância de superfície já corrigida atmosfericamente), acessadas via STAC.

As funções de busca e leitura estão em sentinelUtils.py (ver dependências lá).
"""

import os

import somSeg
import matplotlib.pyplot as plt
import numpy as np

import sentinelUtils

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


bbox = sentinelUtils.criar_bbox(centro_lon, centro_lat, meia_largura_graus)

if os.path.exists(arquivo_cache):
    dados = np.load(arquivo_cache)
    reflectancia, nome_cena = dados["reflectancia"], str(dados["nome_cena"])
else:
    item = sentinelUtils.buscar_cena(bbox, periodo, cobertura_nuvens_max)
    nome_cena = item.id
    print("Cena:", nome_cena, "- nuvens:", item.properties["eo:cloud_cover"], "%")
    reflectancia = sentinelUtils.ler_bandas(item, bbox, bandas)
    np.savez_compressed(arquivo_cache, reflectancia=reflectancia, nome_cena=nome_cena)

print("Imagem:", reflectancia.shape)


# Criando a img de entrada com os canais desejados (bandas + NDVI)
img_shape = reflectancia.shape
numero_canais = img_shape[2] + 1

b04 = reflectancia[:, :, bandas.index("B04")]
b08 = reflectancia[:, :, bandas.index("B08")]
ndvi = sentinelUtils.indice_normalizado(b08, b04)  # Índice de vegetação, entre -1 e 1

ROI_sample_conv = np.zeros((img_shape[0], img_shape[1], numero_canais))
ROI_sample_conv[:, :, 0:img_shape[2]] = reflectancia
ROI_sample_conv[:, :, img_shape[2]] = ndvi
ROI_sample_conv = sentinelUtils.normalizar_0_255(ROI_sample_conv)
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
