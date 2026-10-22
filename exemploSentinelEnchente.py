# -*- coding: utf-8 -*-
"""
Exemplo de mapeamento de enchente comparando imagens Sentinel-2 de antes e
depois do evento, usando o somSeg.

Com uma única data, o SOM separa bem a água, mas não sabe dizer se ela é um rio
ou lago permanente ou se é água da enchente, e a vegetação parcialmente
submersa continua parecendo vegetação. Aqui as duas datas são empilhadas em uma
única imagem multicanal (bandas de antes + bandas de depois), então cada pixel
descreve uma trajetória de mudança e cada unidade do SOM agrupa pixels que
mudaram da mesma forma (ex.: vegetação -> água, água -> água, sem mudança).

Caso: enchente de maio de 2024 em Pelotas - RS. As duas cenas são da mesma
órbita e do mesmo tile (T22HCK), então os pixels estão alinhados.

As funções de busca e leitura estão em sentinelUtils.py (ver dependências lá).
"""

import os

import somSeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import sentinelUtils


# Parâmetros da busca das imagens
centro_lon, centro_lat = -52.3265, -31.7725   # Pelotas - RS
meia_largura_graus = 0.025                     # ~5,5 km de lado
periodo_antes = "2024-03-07/2024-03-07"        # Cena sem nuvens antes da enchente
periodo_depois = "2024-05-26/2024-05-26"       # Data do evento
cobertura_nuvens_max = 10                      # Porcentagem máxima de nuvens na cena

# Bandas usadas: B02=azul, B03=verde, B04=vermelho, B08=infravermelho próximo,
# B11=infravermelho de ondas curtas
bandas = ["B02", "B03", "B04", "B08", "B11"]

arquivo_cache = "sentinel_enchente_cache.npz"  # Evita baixar as imagens novamente


bbox = sentinelUtils.criar_bbox(centro_lon, centro_lat, meia_largura_graus)

if os.path.exists(arquivo_cache):
    dados = np.load(arquivo_cache)
    refl_antes, refl_depois = dados["refl_antes"], dados["refl_depois"]
    nuvens, nomes_cenas = dados["nuvens"], [str(n) for n in dados["nomes_cenas"]]
else:
    item_antes = sentinelUtils.buscar_cena(bbox, periodo_antes, cobertura_nuvens_max)
    item_depois = sentinelUtils.buscar_cena(bbox, periodo_depois, cobertura_nuvens_max)
    nomes_cenas = [item_antes.id, item_depois.id]
    print("Cenas:", nomes_cenas)
    refl_antes = sentinelUtils.ler_bandas(item_antes, bbox, bandas)
    refl_depois = sentinelUtils.ler_bandas(item_depois, bbox, bandas)
    if refl_antes.shape != refl_depois.shape:
        raise RuntimeError("As cenas estão em grades diferentes: escolha cenas do mesmo tile")
    # Pixels com nuvem ou sombra em qualquer uma das datas
    nuvens = sentinelUtils.ler_mascara_nuvens(item_antes, bbox) | sentinelUtils.ler_mascara_nuvens(item_depois, bbox)
    np.savez_compressed(arquivo_cache, refl_antes=refl_antes, refl_depois=refl_depois,
                        nuvens=nuvens, nomes_cenas=nomes_cenas)

print("Imagem:", refl_antes.shape, "- pixels com nuvem: %.1f%%" % (100 * nuvens.mean()))


# Índices espectrais de cada data
def indices(refl):
    b03, b04, b08 = [refl[:, :, bandas.index(b)] for b in ["B03", "B04", "B08"]]
    ndvi = sentinelUtils.indice_normalizado(b08, b04)  # Vegetação
    ndwi = sentinelUtils.indice_normalizado(b03, b08)  # Água (> 0 indica água)
    return ndvi, ndwi

ndvi_antes, ndwi_antes = indices(refl_antes)
ndvi_depois, ndwi_depois = indices(refl_depois)

# Criando a img de entrada bi-temporal: bandas + NDWI de antes e de depois
ROI_sample_conv = np.concatenate((refl_antes, ndwi_antes[:, :, None],
                                  refl_depois, ndwi_depois[:, :, None]), axis=2)
ROI_sample_conv = sentinelUtils.normalizar_0_255(ROI_sample_conv)
numero_canais = ROI_sample_conv.shape[2]
nomes_canais = [b + " antes" for b in bandas + ["NDWI"]] + [b + " depois" for b in bandas + ["NDWI"]]

# Composições em cor verdadeira para visualização (mesmo realce nas duas datas)
def cor_verdadeira(refl):
    return np.clip(refl[:, :, [bandas.index("B04"), bandas.index("B03"), bandas.index("B02")]] / 0.15, 0, 1)

rgb_antes, rgb_depois = cor_verdadeira(refl_antes), cor_verdadeira(refl_depois)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(rgb_antes)
ax[0].set_title("Antes\n" + nomes_cenas[0][:27], fontsize=9)
ax[1].imshow(rgb_depois)
ax[1].set_title("Depois\n" + nomes_cenas[1][:27], fontsize=9)
fig.tight_layout()

# Vizualizando os canais de entrada
plt.figure(figsize=(18, 4))
for i in range(0, numero_canais):
    plt.subplot(2, numero_canais // 2, i + 1)
    plt.imshow(ROI_sample_conv[:, :, i], cmap="gray")
    plt.title(nomes_canais[i], fontsize=8)
    plt.axis("off")

# Parâmetros de ajuste do método
patch_size = 1  # Vizinhança de (2*patch_size+1)x(2*patch_size+1) pixels em torno de cada pixel
som_size = 12  # Número de unidades do SOM
epochs = 3  # Quantidade de épocas do treinamento
dataset_percentage = 0.2  # Porcentagem dos pixels sem nuvem usados no treinamento

# Criando o dataset para treinar o SOM, sem os pixels com nuvem
all_data, imgInputSOM = somSeg.prepare_dataset_multichanel(ROI_sample_conv, patch_size)
validos = np.flatnonzero(~nuvens.ravel())
selecionados = np.random.RandomState(0).choice(validos, int(len(validos) * dataset_percentage), replace=False)
train_data = all_data[selecionados]
del all_data
print("Samples")
print(train_data.shape[0])

# Inicializando e treinando o SOM
SOM = somSeg.create_map(som_size, 1, (2*patch_size+1)*(2*patch_size+1)*numero_canais)
SOM = somSeg.train_SOM(SOM, train_data, epochs=epochs)

# Segmentação: cada pixel recebe o índice da unidade vencedora (BMU)
result = somSeg.evaluate_SOM(SOM, ROI_sample_conv, patch_size)

somSeg.plot_result_multiclass_matplotlib(result, som_size, rgb_depois)

somSeg.plot_result_multiclass_plotly(result, som_size)


# Interpretação das unidades: cada unidade recebe uma classe de mudança a partir
# dos índices médios dos seus pixels nas duas datas
limiar_agua = 0  # NDWI acima disso indica água
limiar_perda_ndvi = 0.15  # Queda de NDVI que indica vegetação afetada pela água
ndvi_min_vegetacao = 0.5  # NDVI mínimo, antes, para considerar o pixel vegetação

classes = ["Sem mudança", "Água permanente", "Área inundada",
           "Vegetação afetada (possível alagamento)", "Água que recuou"]
cores_classes = ["#d9d9d9", "#08306b", "#3fa9f5", "#d95f0e", "#74c476"]

classe_unidade = np.zeros(som_size, dtype=int)
print("\nUnidade | pixels | NDWI antes -> depois | NDVI antes -> depois | classe")
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
    print("%7d | %6d | %5.2f -> %5.2f        | %5.2f -> %5.2f        | %s"
          % (u, m.sum(), wa, wd, va, vd, classes[classe_unidade[u]]))

mapa_mudanca = classe_unidade[result]

# Distância entre as unidades e pixels por unidade, coloridos pela classe
nd_x, nd_y = somSeg.find_ND(SOM['weights'])
hits = somSeg.find_Hits(SOM, ROI_sample_conv, patch_size)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
ax[0].plot(np.arange(som_size - 1) + 0.5, nd_x, marker='o')
ax[0].set_xlim(-0.5, som_size - 0.5)
ax[0].set_title('Distância entre unidades vizinhas')
ax[1].bar(np.arange(som_size), hits, color=[cores_classes[c] for c in classe_unidade], edgecolor='k')
ax[1].set_xlim(-0.5, som_size - 0.5)
ax[1].set_title('Pixels por unidade (hits), coloridos pela classe de mudança')
fig.tight_layout()

# Mapa final de mudança, comparado com a água detectada só na imagem de depois
area_pixel_km2 = 10 * 10 / 1e6
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6.5), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(rgb_depois)
ax[0].set_title("Depois (cor verdadeira)")

agua_so_depois = np.ma.masked_where(ndwi_depois <= limiar_agua, np.ones(result.shape))
ax[1].imshow(rgb_depois)
ax[1].imshow(agua_so_depois, cmap=ListedColormap(["#3fa9f5"]), alpha=0.7)
ax[1].set_title("Só a imagem de depois: água (NDWI > 0)\n%.2f km²"
                % ((ndwi_depois > limiar_agua).sum() * area_pixel_km2))

ax[2].imshow(rgb_depois)
ax[2].imshow(np.ma.masked_where(mapa_mudanca == 0, mapa_mudanca),
             cmap=ListedColormap(cores_classes), vmin=-0.5, vmax=len(classes) - 0.5, alpha=0.75)
ax[2].imshow(np.ma.masked_where(~nuvens, np.ones(result.shape)), cmap=ListedColormap(["white"]), alpha=0.9)
ax[2].set_title("SOM bi-temporal: classes de mudança")
legenda = [Patch(color=cores_classes[c], label="%s: %.2f km²" % (classes[c], ((mapa_mudanca == c) & ~nuvens).sum() * area_pixel_km2))
           for c in range(1, len(classes)) if np.any(classe_unidade == c)]
legenda.append(Patch(facecolor="white", edgecolor="k", label="Nuvem (descartado)"))
ax[2].legend(handles=legenda, loc="upper right", fontsize=8, framealpha=0.9)
fig.tight_layout()

inundada = np.isin(mapa_mudanca, [2, 3]) & ~nuvens
print("\nÁrea total afetada pela enchente (inundada + vegetação afetada): %.2f km²" % (inundada.sum() * area_pixel_km2))

plt.show()
