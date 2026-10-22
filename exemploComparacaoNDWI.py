# -*- coding: utf-8 -*-
"""
Compara o mapeamento de enchente pelo SOM bi-temporal com o uso direto de um
índice espectral: o limiar de NDWI aplicado às imagens de antes e de depois.

No método direto, um pixel é água se NDWI > limiar_agua. Água nas duas datas é
água permanente e água só depois é área inundada. Esse método não tem
equivalente à classe "vegetação afetada" do SOM.

Usa a mesma grade, as mesmas cenas e a mesma pasta de cache de
exemploSentinelGrade.py: se a grade já foi processada, as imagens não são
baixadas de novo. A figura mostra os dois mapas, onde eles concordam e
discordam sobre a água e a área atingida em cada área da grade.

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
from enchenteSOM import AREA_PIXEL_KM2, CLASSES, CORES_CLASSES


# Parâmetros da grade e das imagens (iguais aos de exemploSentinelGrade.py)
centro_lon, centro_lat = -52.3265, -31.7725   # Centro da grade: Pelotas - RS
meia_largura_graus = 0.025                     # Cada área tem ~5,5 km de lado
n_linhas, n_colunas = 3, 3                     # Tamanho da grade (linha 0 = norte)
nuvens_max_area = 20                           # % máxima de nuvens para processar uma área

periodo_antes = "2024-03-07/2024-03-07"        # Cena antes da enchente
periodo_depois = "2024-05-26/2024-05-26"       # Data do evento
cobertura_nuvens_max = 10                      # Porcentagem máxima de nuvens na cena

pasta_cache = "cache_grade"                    # Mesmo cache de exemploSentinelGrade.py
arquivo_figura = "comparacao_som_ndwi.png"

# Parâmetros de ajuste do SOM
patch_size = 1
som_size = 12
epochs = 3
dataset_percentage = 0.2

limiar_agua = 0  # NDWI acima disso indica água, nos dois métodos

# Concordância sobre a água (permanente + inundada) entre os dois métodos
NOMES_CONC = ["Sem água nos dois", "Água nos dois métodos", "Água só no SOM", "Água só no NDWI"]
CORES_CONC = ["#d9d9d9", "#3fa9f5", "#e7298a", "#ffd92f"]


os.makedirs(pasta_cache, exist_ok=True)

bbox_total = enchenteSOM.bbox_grade(centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
item_antes = sentinelUtils.buscar_cena(bbox_total, periodo_antes, cobertura_nuvens_max)
item_depois = sentinelUtils.buscar_cena(bbox_total, periodo_depois, cobertura_nuvens_max)
print("Cenas:", item_antes.id, item_depois.id)

areas_grade = []
for i in range(n_linhas):
    for j in range(n_colunas):
        nome = enchenteSOM.nome_area_grade(i, j, n_linhas, n_colunas)
        bbox = enchenteSOM.bbox_area_grade(i, j, centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
        print("\n== %s (linha %d, coluna %d)" % (nome, i, j))

        arquivo_cache = os.path.join(pasta_cache, "area_%d_%d.npz" % (i, j))
        nuvens = enchenteSOM.ler_nuvens(item_antes, item_depois, bbox, arquivo_cache)
        area = {"nome": nome, "bbox": bbox, "pct_nuvens": 100 * nuvens.mean()}
        areas_grade.append(area)
        if area["pct_nuvens"] > nuvens_max_area:
            print("Pulada: %.1f%% de nuvens" % area["pct_nuvens"])
            continue

        refl_antes, refl_depois, nuvens = enchenteSOM.ler_dados(item_antes, item_depois, bbox, arquivo_cache)

        # SOM bi-temporal
        ROI_sample_conv, nomes_canais = enchenteSOM.montar_entrada(refl_antes, refl_depois)
        SOM, result = enchenteSOM.treinar_e_segmentar(ROI_sample_conv, nuvens, patch_size, som_size,
                                                      epochs, dataset_percentage)
        classe_unidade, tabela = enchenteSOM.classificar_unidades(result, nuvens, refl_antes, refl_depois,
                                                                  som_size, limiar_agua)
        mapa_som = classe_unidade[result]

        # Limiar direto de NDWI
        mapa_ndwi = enchenteSOM.mapa_ndwi(refl_antes, refl_depois, limiar_agua)

        agua_som = np.isin(mapa_som, [1, 2])
        agua_ndwi = np.isin(mapa_ndwi, [1, 2])
        conc = np.zeros(mapa_som.shape, dtype=int)
        conc[agua_som & agua_ndwi] = 1
        conc[agua_som & ~agua_ndwi] = 2
        conc[~agua_som & agua_ndwi] = 3
        validos = ~nuvens

        area.update(mapa_som=mapa_som, mapa_ndwi=mapa_ndwi, conc=conc, nuvens=nuvens,
                    rgb=enchenteSOM.cor_verdadeira(refl_depois),
                    areas_som=enchenteSOM.areas_por_classe(mapa_som, nuvens),
                    areas_ndwi=enchenteSOM.areas_por_classe(mapa_ndwi, nuvens),
                    # Interseção sobre união da água dos dois métodos
                    iou=(agua_som & agua_ndwi & validos).sum() / max(((agua_som | agua_ndwi) & validos).sum(), 1),
                    so_som=((conc == 2) & validos).sum() * AREA_PIXEL_KM2,
                    so_ndwi=((conc == 3) & validos).sum() * AREA_PIXEL_KM2,
                    # Vegetação afetada pelo SOM em que o NDWI não vê água
                    veg_sem_agua_ndwi=((mapa_som == 3) & ~agua_ndwi & validos).sum() * AREA_PIXEL_KM2)
        print("SOM:  permanente %.2f km², inundada %.2f km², vegetação afetada %.2f km²"
              % tuple(area["areas_som"][[1, 2, 3]]))
        print("NDWI: permanente %.2f km², inundada %.2f km², água que recuou %.2f km²"
              % tuple(area["areas_ndwi"][[1, 2, 4]]))
        print("Água SOM x NDWI: IoU %.3f, só no SOM %.2f km², só no NDWI %.2f km²"
              % (area["iou"], area["so_som"], area["so_ndwi"]))

processadas = [a for a in areas_grade if "mapa_som" in a]
if not processadas:
    raise RuntimeError("Nenhuma área processada: todas têm mais de %g%% de nuvens" % nuvens_max_area)


# Os mapas mostram só as linhas e colunas da grade com alguma área processada
xs = [b for a in processadas for b in (a["bbox"][0], a["bbox"][2])]
ys = [b for a in processadas for b in (a["bbox"][1], a["bbox"][3])]
limites = [min(xs), max(xs), min(ys), max(ys)]
desenhar = [a for a in areas_grade
            if a["bbox"][0] >= limites[0] - 1e-9 and a["bbox"][2] <= limites[1] + 1e-9
            and a["bbox"][1] >= limites[2] - 1e-9 and a["bbox"][3] <= limites[3] + 1e-9]

fig = plt.figure(figsize=(19, 21))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75], hspace=0.28, wspace=0.12,
                      left=0.06, right=0.97, top=0.95, bottom=0.04)
ax = [fig.add_subplot(gs[k // 2, k % 2]) for k in range(4)]
titulos = ["Depois - " + item_depois.datetime.strftime("%d/%m/%Y"), "SOM bi-temporal (somSeg)",
           "Limiar direto de NDWI (antes e depois)", "Concordância da água (permanente + inundada)"]

for a in desenhar:
    x0, y0, x1, y1 = a["bbox"]
    ext = [x0, x1, y0, y1]
    for k in range(4):
        if "mapa_som" not in a:
            ax[k].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#bdbdbd", hatch="//", edgecolor="none"))
            ax[k].text((x0 + x1) / 2, (y0 + y1) / 2, "%.0f%% de nuvens\n(não processada)" % a["pct_nuvens"],
                       ha="center", va="center", fontsize=11)
        else:
            ax[k].imshow(a["rgb"], extent=ext)
            if k in [1, 2]:
                mapa = a["mapa_som"] if k == 1 else a["mapa_ndwi"]
                ax[k].imshow(np.ma.masked_where(mapa == 0, mapa), cmap=ListedColormap(CORES_CLASSES),
                             vmin=-0.5, vmax=len(CLASSES) - 0.5, alpha=0.85, extent=ext)
            if k == 3:
                ax[k].imshow(np.ma.masked_where(a["conc"] == 0, a["conc"]), cmap=ListedColormap(CORES_CONC),
                             vmin=-0.5, vmax=len(NOMES_CONC) - 0.5, alpha=0.9, extent=ext)
            if k > 0:
                ax[k].imshow(np.ma.masked_where(~a["nuvens"], np.ones(a["nuvens"].shape)),
                             cmap=ListedColormap(["white"]), alpha=0.9, extent=ext)
        ax[k].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1))
        ax[k].text(x0 + 0.002, y1 - 0.002, a["nome"], color="yellow", fontsize=11, va="top", fontweight="bold",
                   bbox=dict(facecolor="black", alpha=0.4, lw=0, pad=1.5))

for k in range(4):
    ax[k].set_xlim(limites[0], limites[1])
    ax[k].set_ylim(limites[2], limites[3])
    ax[k].set_aspect(1 / np.cos(np.radians(centro_lat)))  # Mantém a proporção em km
    ax[k].set_title(titulos[k], fontsize=13)
    ax[k].tick_params(labelsize=9)
legenda_classes = [Patch(color=CORES_CLASSES[c], label=CLASSES[c]) for c in range(1, len(CLASSES))]
legenda_classes.append(Patch(facecolor="white", edgecolor="k", label="Nuvem (descartado)"))
for k in [1, 2]:
    ax[k].legend(handles=legenda_classes, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)
ax[3].legend(handles=[Patch(color=CORES_CONC[c], label=NOMES_CONC[c]) for c in [1, 2, 3]],
             loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)

# Área atingida pela enchente em cada área, pelos dois métodos
axb = fig.add_subplot(gs[2, 0])
n = np.arange(len(processadas))
w = 0.38
som_inundada = np.array([a["areas_som"][2] for a in processadas])
som_vegetacao = np.array([a["areas_som"][3] for a in processadas])
ndwi_inundada = np.array([a["areas_ndwi"][2] for a in processadas])
axb.barh(n - w / 2, som_inundada, w, color=CORES_CLASSES[2], edgecolor="k", label="SOM: área inundada")
axb.barh(n - w / 2, som_vegetacao, w, left=som_inundada, color=CORES_CLASSES[3], edgecolor="k",
         label="SOM: vegetação afetada")
axb.barh(n + w / 2, ndwi_inundada, w, color=CORES_CLASSES[2], edgecolor="k", hatch="//", label="NDWI: área inundada")
for k in range(len(processadas)):
    axb.text(som_inundada[k] + som_vegetacao[k] + 0.3, k - w / 2, "%.1f" % (som_inundada[k] + som_vegetacao[k]),
             va="center", fontsize=9)
    axb.text(ndwi_inundada[k] + 0.3, k + w / 2, "%.1f" % ndwi_inundada[k], va="center", fontsize=9)
axb.set_yticks(n)
axb.set_yticklabels([a["nome"] for a in processadas])
axb.invert_yaxis()
axb.set_xlim(0, max(som_inundada + som_vegetacao) * 1.12)
axb.set_xlabel("Área (km²)")
axb.set_title("Área atingida pela enchente: SOM x NDWI")
axb.legend(fontsize=9, loc="upper right")
for s in ["top", "right"]:
    axb.spines[s].set_visible(False)

# Tabela resumo
axt = fig.add_subplot(gs[2, 1])
axt.axis("off")
linhas = [[a["nome"], "%.2f" % a["iou"], "%.2f" % a["so_som"], "%.2f" % a["so_ndwi"],
           "%.2f / %.2f" % (a["areas_som"][1], a["areas_ndwi"][1]), "%.2f" % a["veg_sem_agua_ndwi"]]
          for a in processadas]
soma = lambda chave: sum(a[chave] for a in processadas)
linhas.append(["Total", "-", "%.2f" % soma("so_som"), "%.2f" % soma("so_ndwi"),
               "%.2f / %.2f" % (sum(a["areas_som"][1] for a in processadas),
                                sum(a["areas_ndwi"][1] for a in processadas)),
               "%.2f" % soma("veg_sem_agua_ndwi")])
tabela = axt.table(cellText=linhas, cellLoc="center", loc="center",
                   colLabels=["Área", "IoU da água\nSOM x NDWI", "Água só no\nSOM (km²)", "Água só no\nNDWI (km²)",
                              "Água permanente\nSOM / NDWI (km²)", "Veg. afetada sem\nágua no NDWI (km²)"])
tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1, 2.3)
axt.set_title("Comparação por área", fontsize=11)

fig.suptitle("Mapeamento de enchente: SOM bi-temporal x limiar direto de NDWI (NDWI > %g)" % limiar_agua, fontsize=15)
fig.savefig(arquivo_figura, dpi=100)
print("\nFigura salva em", arquivo_figura)

total_som = sum(a["areas_som"][enchenteSOM.CLASSES_ENCHENTE].sum() for a in processadas)
total_ndwi = sum(a["areas_ndwi"][2] for a in processadas)
print("Área atingida nas %d áreas processadas: SOM %.2f km² (inundada + vegetação afetada), NDWI %.2f km²"
      % (len(processadas), total_som, total_ndwi))

plt.show()
