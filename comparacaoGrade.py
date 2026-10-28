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


# Processa uma área com os dois métodos. Se a área tiver mais de nuvens_max_area %
# de pixels inválidos, ela é pulada. Retorna um dicionário com o nome, o bbox e,
# se processada, os mapas e as áreas por classe (as imagens em uint8 e os mapas
# em int8, para ocupar menos memória em grades grandes)
def processar_area(item_antes, item_depois, bbox, nome, arquivo_cache=None, nuvens_max_area=20, patch_size=1,
                   som_size=12, epochs=3, dataset_percentage=0.02, limiar_agua=0):
    nuvens = enchenteSOM.ler_nuvens(item_antes, item_depois, bbox, arquivo_cache)
    area = {"nome": nome, "bbox": bbox, "pct_nuvens": 100 * nuvens.mean()}
    if area["pct_nuvens"] > nuvens_max_area:
        print("%s: pulada, %.1f%% de nuvens ou sem dado" % (nome, area["pct_nuvens"]), flush=True)
        return area

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
    conc = np.zeros(mapa_som.shape, dtype=np.int8)
    conc[agua_som & agua_ndwi] = 1
    conc[agua_som & ~agua_ndwi] = 2
    conc[~agua_som & agua_ndwi] = 3
    validos = ~nuvens

    area.update(mapa_som=mapa_som.astype(np.int8), mapa_ndwi=mapa_ndwi.astype(np.int8), conc=conc, nuvens=nuvens,
                rgb=(enchenteSOM.cor_verdadeira(refl_depois) * 255).astype(np.uint8),
                rgb_antes=(enchenteSOM.cor_verdadeira(refl_antes) * 255).astype(np.uint8),
                areas_som=enchenteSOM.areas_por_classe(mapa_som, nuvens),
                areas_ndwi=enchenteSOM.areas_por_classe(mapa_ndwi, nuvens),
                area_valida=validos.sum() * AREA_PIXEL_KM2,
                # Interseção sobre união da água dos dois métodos
                iou=(agua_som & agua_ndwi & validos).sum() / max(((agua_som | agua_ndwi) & validos).sum(), 1),
                so_som=((conc == 2) & validos).sum() * AREA_PIXEL_KM2,
                so_ndwi=((conc == 3) & validos).sum() * AREA_PIXEL_KM2)
    print("%s: SOM permanente %.2f km², alagamento %.2f km² | NDWI permanente %.2f km², alagamento %.2f km² | "
          "IoU da água %.3f" % (nome, area["areas_som"][1], area["areas_som"][2], area["areas_ndwi"][1],
                                area["areas_ndwi"][2], area["iou"]), flush=True)
    return area

# Processa cada área da grade com os dois métodos. Áreas com mais de
# nuvens_max_area % de pixels inválidos são puladas. Retorna uma lista com um
# dicionário por área (as processadas têm os mapas e as áreas por classe)
def processar_grade(item_antes, item_depois, centro_lon, centro_lat, meia_largura_graus,
                    n_linhas, n_colunas, pasta_cache, nuvens_max_area=20, patch_size=1,
                    som_size=12, epochs=3, dataset_percentage=0.02, limiar_agua=0):
    os.makedirs(pasta_cache, exist_ok=True)
    areas_grade = []
    for i in range(n_linhas):
        for j in range(n_colunas):
            nome = enchenteSOM.nome_area_grade(i, j, n_linhas, n_colunas)
            bbox = enchenteSOM.bbox_area_grade(i, j, centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
            arquivo_cache = os.path.join(pasta_cache, "area_%d_%d.npz" % (i, j))
            areas_grade.append(processar_area(item_antes, item_depois, bbox, nome, arquivo_cache, nuvens_max_area,
                                              patch_size, som_size, epochs, dataset_percentage, limiar_agua))
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

# Figura com as imagens de antes e de depois, os mapas dos dois métodos e a
# concordância da água. Um painel reúne as legendas e os totais. Em grades
# grandes, mostrar_nomes=False esconde o nome e os textos de cada área
def figura_comparacao(areas_grade, item_antes, item_depois, centro_lat, limiar_agua=0, titulo=None,
                      arquivo=None, dpi=100, mostrar_nomes=True, rotulo_depois=None):
    processadas = [a for a in areas_grade if "mapa_som" in a]

    # Os mapas mostram só as linhas e colunas da grade com alguma área processada
    base = processadas if processadas else areas_grade
    xs = [b for a in base for b in (a["bbox"][0], a["bbox"][2])]
    ys = [b for a in base for b in (a["bbox"][1], a["bbox"][3])]
    limites = [min(xs), max(xs), min(ys), max(ys)]
    desenhar = [a for a in areas_grade
                if a["bbox"][0] >= limites[0] - 1e-9 and a["bbox"][2] <= limites[1] + 1e-9
                and a["bbox"][1] >= limites[2] - 1e-9 and a["bbox"][3] <= limites[3] + 1e-9]

    # Linha de cima: antes, depois e painel de legendas; linha de baixo: SOM, NDWI e concordância.
    # A altura da figura acompanha a proporção da região, para regiões largas não sobrar espaço
    proporcao = (limites[3] - limites[2]) / ((limites[1] - limites[0]) * np.cos(np.radians(centro_lat)))
    fig, eixos = plt.subplots(nrows=2, ncols=3, figsize=(24, float(np.clip(14 * proporcao, 9, 16.5))))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.04, wspace=0.15, hspace=0.12)
    ax_antes, ax_depois, ax_info = eixos[0]
    ax_som, ax_ndwi, ax_conc = eixos[1]
    mapas = [ax_antes, ax_depois, ax_som, ax_ndwi, ax_conc]
    titulos = ["Antes: " + item_antes.datetime.strftime("%d/%m/%Y"),
               rotulo_depois or "Depois: " + item_depois.datetime.strftime("%d/%m/%Y"),
               "SOM bi-temporal (somSeg, unidades classificadas pelo MNDWI)",
               "Baseline: limiar direto de NDWI (antes e depois)",
               "Concordância da água (permanente + alagamento)"]

    for a in desenhar:
        x0, y0, x1, y1 = a["bbox"]
        ext = [x0, x1, y0, y1]
        for k, ax in enumerate(mapas):
            if "mapa_som" not in a:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#bdbdbd", hatch="//", edgecolor="none"))
                if mostrar_nomes:
                    ax.text((x0 + x1) / 2, (y0 + y1) / 2, "%.0f%% de nuvens\nou sem dado\n(não processada)"
                            % a["pct_nuvens"], ha="center", va="center", fontsize=10)
            else:
                ax.imshow(a["rgb_antes"] if k == 0 else a["rgb"], extent=ext)
                if k in [2, 3]:
                    mapa = a["mapa_som"] if k == 2 else a["mapa_ndwi"]
                    ax.imshow(np.ma.masked_where(mapa == 0, mapa), cmap=ListedColormap(CORES_CLASSES),
                              vmin=-0.5, vmax=len(CLASSES) - 0.5, alpha=0.85, extent=ext)
                if k == 4:
                    ax.imshow(np.ma.masked_where(a["conc"] == 0, a["conc"]), cmap=ListedColormap(CORES_CONC),
                              vmin=-0.5, vmax=len(NOMES_CONC) - 0.5, alpha=0.9, extent=ext)
                if k >= 2:
                    ax.imshow(np.ma.masked_where(~a["nuvens"], np.ones(a["nuvens"].shape)),
                              cmap=ListedColormap(["white"]), alpha=0.9, extent=ext)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow",
                                   lw=1 if mostrar_nomes else 0.4))
            if mostrar_nomes:
                ax.text(x0 + 0.002, y1 - 0.002, a["nome"], color="yellow", fontsize=10, va="top",
                        fontweight="bold", bbox=dict(facecolor="black", alpha=0.4, lw=0, pad=1.5))

    for ax, t in zip(mapas, titulos):
        ax.set_xlim(limites[0], limites[1])
        ax.set_ylim(limites[2], limites[3])
        ax.set_aspect(1 / np.cos(np.radians(centro_lat)))  # Mantém a proporção em km
        ax.set_title(t, fontsize=13)
        ax.tick_params(labelsize=9)

    # Painel com as legendas e os totais
    ax_info.axis("off")
    legenda_classes = [Patch(color=CORES_CLASSES[c], label=CLASSES[c]) for c in range(1, len(CLASSES))]
    legenda_classes.append(Patch(facecolor="white", edgecolor="k", label="Nuvem ou sem dado (descartado)"))
    # Legendas lado a lado no topo e os totais embaixo
    l1 = ax_info.legend(handles=legenda_classes, title="Classes (SOM e NDWI)", loc="upper left",
                        bbox_to_anchor=(0.0, 1.0), fontsize=11, title_fontsize=12, frameon=False)
    ax_info.add_artist(l1)
    ax_info.legend(handles=[Patch(color=CORES_CONC[c], label=NOMES_CONC[c]) for c in [1, 2, 3]],
                   title="Concordância da água", loc="upper left", bbox_to_anchor=(0.56, 1.0),
                   fontsize=11, title_fontsize=12, frameon=False)
    if processadas:
        t = totais(areas_grade)
        linha = "%-22s%8s%8s\n"
        texto = ("Áreas processadas: %d de %d (%.1f km² válidos)\n\n"
                 % (t["areas_processadas"], len(areas_grade), t["area_valida_km2"])
                 + linha % ("", "SOM", "NDWI")
                 + linha % ("Alagamento (km²)", "%.1f" % t["som_alagamento_km2"], "%.1f" % t["ndwi_alagamento_km2"])
                 + linha % ("Água permanente (km²)", "%.1f" % t["som_permanente_km2"], "%.1f" % t["ndwi_permanente_km2"])
                 + "\nIoU da água SOM x NDWI: %.2f" % t["iou_agua"])
    else:
        texto = "Nenhuma área processada (nuvens ou sem dado)"
    ax_info.text(0.02, 0.5, texto, fontsize=12, family="monospace", va="top", transform=ax_info.transAxes)

    if titulo is None:
        titulo = "Mapeamento de enchente: SOM bi-temporal x limiar direto de NDWI (NDWI > %g)" % limiar_agua
    fig.suptitle(titulo, fontsize=16)
    if arquivo is not None:
        fig.savefig(arquivo, dpi=dpi)
    return fig
