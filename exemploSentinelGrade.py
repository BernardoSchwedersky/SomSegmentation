# -*- coding: utf-8 -*-
"""
Mapeamento de enchente em uma grade de áreas vizinhas, com o SOM bi-temporal
de exemploSentinelEnchente.py aplicado a cada área.

A região é dividida em uma grade de n_linhas x n_colunas áreas do mesmo
tamanho, centrada em um ponto. As mesmas duas cenas (antes e depois) são usadas
para todas as áreas, e cada área tem o seu próprio SOM. Áreas com muitas nuvens
em qualquer uma das datas são puladas. O resultado é um mosaico com as imagens
de antes e depois, o mapa de classes de mudança e a área de cada classe por
área da grade.

Caso: enchente de maio de 2024 em Pelotas - RS.

As etapas do método estão em enchenteSOM.py e as funções de busca e leitura em
sentinelUtils.py (ver dependências lá).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

import enchenteSOM
import sentinelUtils
from enchenteSOM import CLASSES, CORES_CLASSES


# Parâmetros da grade
centro_lon, centro_lat = -52.3265, -31.7725   # Centro da grade: Pelotas - RS
meia_largura_graus = 0.025                     # Cada área tem ~5,5 km de lado
n_linhas, n_colunas = 3, 3                     # Tamanho da grade (linha 0 = norte)
nuvens_max_area = 20                           # % máxima de nuvens para processar uma área

# Parâmetros da busca das imagens
periodo_antes = "2024-03-07/2024-03-07"        # Cena antes da enchente
periodo_depois = "2024-05-26/2024-05-26"       # Data do evento
cobertura_nuvens_max = 10                      # Porcentagem máxima de nuvens na cena

pasta_cache = "cache_grade"                    # Evita baixar as imagens novamente
arquivo_figura = "enchente_grade.png"

# Parâmetros de ajuste do método (iguais aos de exemploSentinelEnchente.py)
patch_size = 1
som_size = enchenteSOM.SOM_SIZE
epochs = 3
dataset_percentage = 0.02


# Retângulo e nome de cada área da grade
def bbox_area(i, j):
    return enchenteSOM.bbox_area_grade(i, j, centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)

def nome_area(i, j):
    return enchenteSOM.nome_area_grade(i, j, n_linhas, n_colunas)


os.makedirs(pasta_cache, exist_ok=True)

# As mesmas cenas são usadas em toda a grade, para os pixels ficarem alinhados
bbox_grade = enchenteSOM.bbox_grade(centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
item_antes = sentinelUtils.buscar_cena(bbox_grade, periodo_antes, cobertura_nuvens_max)
item_depois = sentinelUtils.buscar_cena(bbox_grade, periodo_depois, cobertura_nuvens_max)
print("Cenas:", item_antes.id, item_depois.id)

areas_grade = []
for i in range(n_linhas):
    for j in range(n_colunas):
        nome = nome_area(i, j)
        bbox = bbox_area(i, j)
        print("\n== %s (linha %d, coluna %d)" % (nome, i, j))

        # Confere as nuvens antes de baixar as bandas
        arquivo_cache = os.path.join(pasta_cache, "area_%d_%d.npz" % (i, j))
        nuvens = enchenteSOM.ler_nuvens(item_antes, item_depois, bbox, arquivo_cache)
        pct_nuvens = 100 * nuvens.mean()
        area = {"nome": nome, "bbox": bbox, "linha": i, "coluna": j, "pct_nuvens": pct_nuvens}
        areas_grade.append(area)
        if pct_nuvens > nuvens_max_area:
            print("Pulada: %.1f%% de nuvens" % pct_nuvens)
            continue

        refl_antes, refl_depois, nuvens = enchenteSOM.ler_dados(item_antes, item_depois, bbox, arquivo_cache)
        ROI_sample_conv, nomes_canais = enchenteSOM.montar_entrada(refl_antes, refl_depois)
        SOM, result = enchenteSOM.treinar_e_segmentar(ROI_sample_conv, nuvens, patch_size, som_size,
                                                      epochs, dataset_percentage)
        classe_unidade, tabela = enchenteSOM.classificar_unidades(result, nuvens, refl_antes, refl_depois, som_size)
        enchenteSOM.imprimir_tabela(tabela)

        area["mapa"] = classe_unidade[result]
        area["nuvens"] = nuvens
        area["rgb_antes"] = enchenteSOM.cor_verdadeira(refl_antes)
        area["rgb_depois"] = enchenteSOM.cor_verdadeira(refl_depois)
        area["areas"] = enchenteSOM.areas_por_classe(area["mapa"], nuvens)
        area["area_total"] = area["mapa"].size * enchenteSOM.AREA_PIXEL_KM2
        print("Pixels com nuvem: %.1f%% - alagamento: %.2f km²"
              % (pct_nuvens, area["areas"][enchenteSOM.CLASSES_ENCHENTE].sum()))

processadas = [a for a in areas_grade if "mapa" in a]


# Mosaico: cada área é desenhada na sua posição geográfica
fig = plt.figure(figsize=(19, 7 + 0.45 * len(processadas)))
gs = fig.add_gridspec(2, 3, height_ratios=[6, 0.45 * len(processadas) + 1])
ax = [fig.add_subplot(gs[0, k]) for k in range(3)]
for a in areas_grade:
    x0, y0, x1, y1 = a["bbox"]
    ext = [x0, x1, y0, y1]
    for k in range(3):
        if "mapa" not in a:
            ax[k].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#bdbdbd", hatch="//", edgecolor="none"))
            ax[k].text((x0 + x1) / 2, (y0 + y1) / 2, "%.0f%% de nuvens\n(não processada)" % a["pct_nuvens"],
                       ha="center", va="center", fontsize=9)
            continue
        ax[k].imshow(a["rgb_depois"] if k > 0 else a["rgb_antes"], extent=ext)
        if k == 2:
            mapa = a["mapa"]
            ax[k].imshow(np.ma.masked_where(mapa == 0, mapa), cmap=ListedColormap(CORES_CLASSES),
                         vmin=-0.5, vmax=len(CLASSES) - 0.5, alpha=0.8, extent=ext)
            ax[k].imshow(np.ma.masked_where(~a["nuvens"], np.ones(mapa.shape)),
                         cmap=ListedColormap(["white"]), alpha=0.9, extent=ext)
    for k in range(3):
        ax[k].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1.2))
        ax[k].text(x0 + 0.002, y1 - 0.002, a["nome"], color="yellow", fontsize=9, va="top", fontweight="bold",
                   bbox=dict(facecolor="black", alpha=0.4, lw=0, pad=1.5))

ax[0].set_title("Antes - " + item_antes.datetime.strftime("%d/%m/%Y"))
ax[1].set_title("Depois - " + item_depois.datetime.strftime("%d/%m/%Y"))
ax[2].set_title("SOM bi-temporal: classes de mudança (um SOM por área)")
for k in range(3):
    ax[k].set_xlim(bbox_grade[0], bbox_grade[2])
    ax[k].set_ylim(bbox_grade[1], bbox_grade[3])
    ax[k].set_aspect(1 / np.cos(np.radians(centro_lat)))  # Mantém a proporção em km
    ax[k].set_xlabel("Longitude")
    ax[k].set_ylabel("Latitude")
    ax[k].tick_params(labelsize=8)
presentes = [c for c in range(1, len(CLASSES)) if any(a["areas"][c] > 0 for a in processadas)]
legenda = [Patch(color=CORES_CLASSES[c], label=CLASSES[c]) for c in presentes]
legenda.append(Patch(facecolor="white", edgecolor="k", label="Nuvem (descartado)"))
ax[2].legend(handles=legenda, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

# Área de cada classe por área da grade
axb = fig.add_subplot(gs[1, :])
if processadas:
    nomes = [a["nome"] for a in processadas]
    base = np.zeros(len(processadas))
    for c in range(1, len(CLASSES)):
        v = np.array([a["areas"][c] for a in processadas])
        if not np.any(v > 0):
            continue
        axb.barh(nomes, v, left=base, color=CORES_CLASSES[c], edgecolor="k", label=CLASSES[c])
        for k, x in enumerate(v):
            if x > 1:
                axb.text(base[k] + x / 2, k, "%.1f" % x, ha="center", va="center", fontsize=9,
                         color="white" if c in [1, 2] else "black")
        base += v
    total_max = max(a["area_total"] for a in processadas)
    for k, a in enumerate(processadas):
        atingida = a["areas"][enchenteSOM.CLASSES_ENCHENTE].sum()
        axb.text(base[k] + total_max * 0.01, k, "alagamento: %.1f km² (%.0f%% de %.1f km²)"
                 % (atingida, 100 * atingida / a["area_total"], a["area_total"]), va="center", fontsize=9)
    axb.set_xlim(0, total_max * 1.35)
    axb.invert_yaxis()
    axb.legend(loc="lower right", fontsize=8)
axb.set_xlabel("Área (km²)")
axb.set_title("Área por classe em cada área da grade")
for s in ["top", "right"]:
    axb.spines[s].set_visible(False)

fig.suptitle("Mapeamento de enchente com Sentinel-2 + SOM bi-temporal (somSeg)", fontsize=14)
fig.tight_layout()
fig.savefig(arquivo_figura, dpi=110)
print("\nFigura salva em", arquivo_figura)

total = sum(a["areas"][enchenteSOM.CLASSES_ENCHENTE].sum() for a in processadas)
print("Alagamento nas %d áreas processadas: %.2f km²" % (len(processadas), total))

plt.show()
