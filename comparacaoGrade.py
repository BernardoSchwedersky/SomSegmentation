# -*- coding: utf-8 -*-
"""
Processamento de uma grade de áreas com o SOM bi-temporal e com o limiar direto
de NDWI (baseline), e a figura comparando os dois métodos. Usado em
exemploComparacaoNDWI.py (um evento) e processarEventos.py (vários eventos).

Os dois métodos usam as mesmas três classes: água nas duas datas é água
permanente, água só depois é alagamento e o resto é não alagamento. No SOM a
classe é atribuída a cada unidade pelo MNDWI médio dos seus pixels; no
baseline cada pixel é água se NDWI > limiar_agua.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

import enchenteSOM
from enchenteSOM import AREA_PIXEL_KM2, CLASSES, CORES_CLASSES


# Concordância sobre a água (permanente + alagamento) entre os dois métodos
NOMES_CONC = ["Sem água nos dois", "Água nos dois métodos", "Água só no SOM", "Água só no NDWI"]
CORES_CONC = ["#d9d9d9", "#3fa9f5", "#e7298a", "#ffd92f"]


# Processa cada área da grade com os dois métodos. Áreas com mais de
# nuvens_max_area % de pixels inválidos são puladas. Retorna uma lista com um
# dicionário por área (as processadas têm os mapas e as áreas por classe)
def processar_grade(item_antes, item_depois, centro_lon, centro_lat, meia_largura_graus,
                    n_linhas, n_colunas, pasta_cache, nuvens_max_area=20, patch_size=1,
                    som_size=12, epochs=3, dataset_percentage=0.2, limiar_agua=0):
    os.makedirs(pasta_cache, exist_ok=True)
    areas_grade = []
    for i in range(n_linhas):
        for j in range(n_colunas):
            nome = enchenteSOM.nome_area_grade(i, j, n_linhas, n_colunas)
            bbox = enchenteSOM.bbox_area_grade(i, j, centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
            print("\n== %s (linha %d, coluna %d)" % (nome, i, j), flush=True)

            arquivo_cache = os.path.join(pasta_cache, "area_%d_%d.npz" % (i, j))
            nuvens = enchenteSOM.ler_nuvens(item_antes, item_depois, bbox, arquivo_cache)
            area = {"nome": nome, "bbox": bbox, "pct_nuvens": 100 * nuvens.mean()}
            areas_grade.append(area)
            if area["pct_nuvens"] > nuvens_max_area:
                print("Pulada: %.1f%% de nuvens ou sem dado" % area["pct_nuvens"])
                continue

            refl_antes, refl_depois, nuvens = enchenteSOM.ler_dados(item_antes, item_depois, bbox, arquivo_cache)

            # SOM bi-temporal
            ROI_sample_conv, nomes_canais = enchenteSOM.montar_entrada(refl_antes, refl_depois)
            SOM, result = enchenteSOM.treinar_e_segmentar(ROI_sample_conv, nuvens, patch_size, som_size,
                                                          epochs, dataset_percentage)
            classe_unidade, tabela = enchenteSOM.classificar_unidades(result, nuvens, refl_antes, refl_depois,
                                                                      som_size, limiar_agua)
            mapa_som = classe_unidade[result]

            # Baseline: limiar direto de NDWI
            mapa_ndwi = enchenteSOM.mapa_indice(refl_antes, refl_depois, limiar_agua)

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
                        area_valida=validos.sum() * AREA_PIXEL_KM2,
                        # Interseção sobre união da água dos dois métodos
                        iou=(agua_som & agua_ndwi & validos).sum() / max(((agua_som | agua_ndwi) & validos).sum(), 1),
                        so_som=((conc == 2) & validos).sum() * AREA_PIXEL_KM2,
                        so_ndwi=((conc == 3) & validos).sum() * AREA_PIXEL_KM2)
            print("SOM:  água permanente %.2f km², alagamento %.2f km²" % tuple(area["areas_som"][[1, 2]]))
            print("NDWI: água permanente %.2f km², alagamento %.2f km²" % tuple(area["areas_ndwi"][[1, 2]]))
            print("Água SOM x NDWI: IoU %.3f, só no SOM %.2f km², só no NDWI %.2f km²"
                  % (area["iou"], area["so_som"], area["so_ndwi"]), flush=True)
    return areas_grade

# Totais das áreas processadas da grade (km²)
def totais(areas_grade):
    proc = [a for a in areas_grade if "mapa_som" in a]
    soma = lambda f: float(sum(f(a) for a in proc))
    agua_som = soma(lambda a: a["areas_som"][1] + a["areas_som"][2])
    agua_ndwi = soma(lambda a: a["areas_ndwi"][1] + a["areas_ndwi"][2])
    so_som, so_ndwi = soma(lambda a: a["so_som"]), soma(lambda a: a["so_ndwi"])
    ambos = agua_som - so_som
    return {"areas_processadas": len(proc), "areas_puladas": len(areas_grade) - len(proc),
            "area_valida_km2": soma(lambda a: a["area_valida"]),
            "som_permanente_km2": soma(lambda a: a["areas_som"][1]),
            "som_alagamento_km2": soma(lambda a: a["areas_som"][2]),
            "ndwi_permanente_km2": soma(lambda a: a["areas_ndwi"][1]),
            "ndwi_alagamento_km2": soma(lambda a: a["areas_ndwi"][2]),
            "agua_so_som_km2": so_som, "agua_so_ndwi_km2": so_ndwi,
            "iou_agua": ambos / max(ambos + so_som + so_ndwi, 1e-9)}

# Figura com os mapas dos dois métodos, a concordância da água, a área de alagamento
# por área e uma tabela resumo
def figura_comparacao(areas_grade, item_antes, item_depois, centro_lat, limiar_agua=0, titulo=None,
                      arquivo=None, dpi=100):
    processadas = [a for a in areas_grade if "mapa_som" in a]

    # Os mapas mostram só as linhas e colunas da grade com alguma área processada
    base = processadas if processadas else areas_grade
    xs = [b for a in base for b in (a["bbox"][0], a["bbox"][2])]
    ys = [b for a in base for b in (a["bbox"][1], a["bbox"][3])]
    limites = [min(xs), max(xs), min(ys), max(ys)]
    desenhar = [a for a in areas_grade
                if a["bbox"][0] >= limites[0] - 1e-9 and a["bbox"][2] <= limites[1] + 1e-9
                and a["bbox"][1] >= limites[2] - 1e-9 and a["bbox"][3] <= limites[3] + 1e-9]

    fig = plt.figure(figsize=(19, 21))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75], hspace=0.28, wspace=0.12,
                          left=0.06, right=0.97, top=0.95, bottom=0.04)
    ax = [fig.add_subplot(gs[k // 2, k % 2]) for k in range(4)]
    titulos = ["Depois: %s (antes: %s)" % (item_depois.datetime.strftime("%d/%m/%Y"),
                                           item_antes.datetime.strftime("%d/%m/%Y")),
               "SOM bi-temporal (somSeg, unidades classificadas pelo MNDWI)",
               "Baseline: limiar direto de NDWI (antes e depois)",
               "Concordância da água (permanente + alagamento)"]

    for a in desenhar:
        x0, y0, x1, y1 = a["bbox"]
        ext = [x0, x1, y0, y1]
        for k in range(4):
            if "mapa_som" not in a:
                ax[k].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#bdbdbd", hatch="//",
                                          edgecolor="none"))
                ax[k].text((x0 + x1) / 2, (y0 + y1) / 2, "%.0f%% de nuvens\nou sem dado\n(não processada)"
                           % a["pct_nuvens"], ha="center", va="center", fontsize=11)
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
            ax[k].text(x0 + 0.002, y1 - 0.002, a["nome"], color="yellow", fontsize=11, va="top",
                       fontweight="bold", bbox=dict(facecolor="black", alpha=0.4, lw=0, pad=1.5))

    for k in range(4):
        ax[k].set_xlim(limites[0], limites[1])
        ax[k].set_ylim(limites[2], limites[3])
        ax[k].set_aspect(1 / np.cos(np.radians(centro_lat)))  # Mantém a proporção em km
        ax[k].set_title(titulos[k], fontsize=13)
        ax[k].tick_params(labelsize=9)
    legenda_classes = [Patch(color=CORES_CLASSES[c], label=CLASSES[c]) for c in range(1, len(CLASSES))]
    legenda_classes.append(Patch(facecolor="white", edgecolor="k", label="Nuvem ou sem dado (descartado)"))
    for k in [1, 2]:
        ax[k].legend(handles=legenda_classes, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)
    ax[3].legend(handles=[Patch(color=CORES_CONC[c], label=NOMES_CONC[c]) for c in [1, 2, 3]],
                 loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)

    # Área de alagamento em cada área, pelos dois métodos
    axb = fig.add_subplot(gs[2, 0])
    axt = fig.add_subplot(gs[2, 1])
    axt.axis("off")
    if processadas:
        n = np.arange(len(processadas))
        w = 0.38
        som_alagamento = np.array([a["areas_som"][2] for a in processadas])
        ndwi_alagamento = np.array([a["areas_ndwi"][2] for a in processadas])
        axb.barh(n - w / 2, som_alagamento, w, color=CORES_CLASSES[2], edgecolor="k", label="SOM")
        axb.barh(n + w / 2, ndwi_alagamento, w, color="white", edgecolor=CORES_CLASSES[2], hatch="//",
                 label="NDWI (baseline)")
        maximo = max(max(som_alagamento), max(ndwi_alagamento), 1)
        for k in range(len(processadas)):
            axb.text(som_alagamento[k] + maximo * 0.01, k - w / 2, "%.1f" % som_alagamento[k], va="center", fontsize=9)
            axb.text(ndwi_alagamento[k] + maximo * 0.01, k + w / 2, "%.1f" % ndwi_alagamento[k], va="center", fontsize=9)
        axb.set_yticks(n)
        axb.set_yticklabels([a["nome"] for a in processadas])
        axb.invert_yaxis()
        axb.set_xlim(0, maximo * 1.12)
        axb.legend(fontsize=9, loc="best")

        # Tabela resumo
        t = totais(areas_grade)
        linhas = [[a["nome"], "%.2f" % a["iou"], "%.2f" % a["so_som"], "%.2f" % a["so_ndwi"],
                   "%.2f / %.2f" % (a["areas_som"][1], a["areas_ndwi"][1]),
                   "%.2f / %.2f" % (a["areas_som"][2], a["areas_ndwi"][2])] for a in processadas]
        linhas.append(["Total", "%.2f" % t["iou_agua"], "%.2f" % t["agua_so_som_km2"], "%.2f" % t["agua_so_ndwi_km2"],
                       "%.2f / %.2f" % (t["som_permanente_km2"], t["ndwi_permanente_km2"]),
                       "%.2f / %.2f" % (t["som_alagamento_km2"], t["ndwi_alagamento_km2"])])
        tabela = axt.table(cellText=linhas, cellLoc="center", loc="center",
                           colLabels=["Área", "IoU da água\nSOM x NDWI", "Água só no\nSOM (km²)",
                                      "Água só no\nNDWI (km²)", "Água permanente\nSOM / NDWI (km²)",
                                      "Alagamento\nSOM / NDWI (km²)"])
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)
        tabela.scale(1, 2.3 if len(linhas) <= 7 else 1.9)
        axt.set_title("Comparação por área (alagamento: SOM %.1f km², NDWI %.1f km²)"
                      % (t["som_alagamento_km2"], t["ndwi_alagamento_km2"]), fontsize=11)
    else:
        axb.text(0.5, 0.5, "Nenhuma área processada", ha="center", va="center", transform=axb.transAxes)
    axb.set_xlabel("Área (km²)")
    axb.set_title("Área de alagamento: SOM x NDWI")
    for s in ["top", "right"]:
        axb.spines[s].set_visible(False)

    if titulo is None:
        titulo = "Mapeamento de enchente: SOM bi-temporal x limiar direto de NDWI (NDWI > %g)" % limiar_agua
    fig.suptitle(titulo, fontsize=15)
    if arquivo is not None:
        fig.savefig(arquivo, dpi=dpi)
    return fig
