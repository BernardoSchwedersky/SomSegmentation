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

As etapas do método estão em enchenteSOM.py e as funções de busca e leitura em
sentinelUtils.py (ver dependências lá). Para processar várias áreas vizinhas,
veja exemploSentinelGrade.py.
"""

import somSeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import enchenteSOM
import sentinelUtils
from enchenteSOM import CLASSES, CORES_CLASSES


# Parâmetros da busca das imagens
centro_lon, centro_lat = -52.3265, -31.7725   # Pelotas - RS
meia_largura_graus = 0.025                     # ~5,5 km de lado
periodo_antes = "2024-03-07/2024-03-07"        # Cena sem nuvens antes da enchente
periodo_depois = "2024-05-26/2024-05-26"       # Data do evento
cobertura_nuvens_max = 10                      # Porcentagem máxima de nuvens na cena

arquivo_cache = "sentinel_enchente_cache.npz"  # Evita baixar as imagens novamente

# Parâmetros de ajuste do método
patch_size = 1  # Vizinhança de (2*patch_size+1)x(2*patch_size+1) pixels em torno de cada pixel
som_size = 12  # Número de unidades do SOM
epochs = 3  # Quantidade de épocas do treinamento
dataset_percentage = 0.2  # Porcentagem dos pixels sem nuvem usados no treinamento

# Limiares da interpretação das unidades
limiar_agua = 0  # NDWI acima disso indica água
limiar_perda_ndvi = 0.15  # Queda de NDVI que indica vegetação afetada pela água
ndvi_min_vegetacao = 0.5  # NDVI mínimo, antes, para considerar o pixel vegetação


bbox = sentinelUtils.criar_bbox(centro_lon, centro_lat, meia_largura_graus)

item_antes = sentinelUtils.buscar_cena(bbox, periodo_antes, cobertura_nuvens_max)
item_depois = sentinelUtils.buscar_cena(bbox, periodo_depois, cobertura_nuvens_max)
nomes_cenas = [item_antes.id, item_depois.id]
print("Cenas:", nomes_cenas)

refl_antes, refl_depois, nuvens = enchenteSOM.ler_dados(item_antes, item_depois, bbox, arquivo_cache)
print("Imagem:", refl_antes.shape, "- pixels com nuvem: %.1f%%" % (100 * nuvens.mean()))

# Criando a img de entrada bi-temporal: bandas + NDWI de antes e de depois
ROI_sample_conv, nomes_canais = enchenteSOM.montar_entrada(refl_antes, refl_depois)
numero_canais = ROI_sample_conv.shape[2]

rgb_antes, rgb_depois = enchenteSOM.cor_verdadeira(refl_antes), enchenteSOM.cor_verdadeira(refl_depois)

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

# Treinando o SOM (sem os pixels com nuvem) e segmentando a imagem
SOM, result = enchenteSOM.treinar_e_segmentar(ROI_sample_conv, nuvens, patch_size, som_size,
                                              epochs, dataset_percentage)

somSeg.plot_result_multiclass_matplotlib(result, som_size, rgb_depois)

somSeg.plot_result_multiclass_plotly(result, som_size)

# Interpretação das unidades: cada unidade recebe uma classe de mudança a partir
# dos índices médios dos seus pixels nas duas datas
classe_unidade, tabela = enchenteSOM.classificar_unidades(
    result, nuvens, refl_antes, refl_depois, som_size,
    limiar_agua, limiar_perda_ndvi, ndvi_min_vegetacao)
print()
enchenteSOM.imprimir_tabela(tabela)

mapa_mudanca = classe_unidade[result]
areas = enchenteSOM.areas_por_classe(mapa_mudanca, nuvens)

# Distância entre as unidades e pixels por unidade, coloridos pela classe
nd_x, nd_y = somSeg.find_ND(SOM['weights'])
hits = somSeg.find_Hits(SOM, ROI_sample_conv, patch_size)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
ax[0].plot(np.arange(som_size - 1) + 0.5, nd_x, marker='o')
ax[0].set_xlim(-0.5, som_size - 0.5)
ax[0].set_title('Distância entre unidades vizinhas')
ax[1].bar(np.arange(som_size), hits, color=[CORES_CLASSES[c] for c in classe_unidade], edgecolor='k')
ax[1].set_xlim(-0.5, som_size - 0.5)
ax[1].set_title('Pixels por unidade (hits), coloridos pela classe de mudança')
fig.tight_layout()

# Mapa final de mudança, comparado com a água detectada só na imagem de depois
ndwi_depois = enchenteSOM.indices(refl_depois)[1]
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6.5), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(rgb_depois)
ax[0].set_title("Depois (cor verdadeira)")

agua_so_depois = np.ma.masked_where(ndwi_depois <= limiar_agua, np.ones(result.shape))
ax[1].imshow(rgb_depois)
ax[1].imshow(agua_so_depois, cmap=ListedColormap(["#3fa9f5"]), alpha=0.7)
ax[1].set_title("Só a imagem de depois: água (NDWI > 0)\n%.2f km²"
                % ((ndwi_depois > limiar_agua).sum() * enchenteSOM.AREA_PIXEL_KM2))

ax[2].imshow(rgb_depois)
ax[2].imshow(np.ma.masked_where(mapa_mudanca == 0, mapa_mudanca),
             cmap=ListedColormap(CORES_CLASSES), vmin=-0.5, vmax=len(CLASSES) - 0.5, alpha=0.75)
ax[2].imshow(np.ma.masked_where(~nuvens, np.ones(result.shape)), cmap=ListedColormap(["white"]), alpha=0.9)
ax[2].set_title("SOM bi-temporal: classes de mudança")
legenda = [Patch(color=CORES_CLASSES[c], label="%s: %.2f km²" % (CLASSES[c], areas[c]))
           for c in range(1, len(CLASSES)) if np.any(classe_unidade == c)]
legenda.append(Patch(facecolor="white", edgecolor="k", label="Nuvem (descartado)"))
ax[2].legend(handles=legenda, loc="upper right", fontsize=8, framealpha=0.9)
fig.tight_layout()

print("\nÁrea total afetada pela enchente (inundada + vegetação afetada): %.2f km²"
      % areas[enchenteSOM.CLASSES_ENCHENTE].sum())

plt.show()
