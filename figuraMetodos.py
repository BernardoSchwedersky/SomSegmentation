# -*- coding: utf-8 -*-
"""
Compara o SOM bi-temporal com o limiar direto de NDWI e de MNDWI em uma área de
cada evento do dataset.

Para cada pasta de dataset/, escolhe entre as áreas a no máximo dist_max áreas
do centro do evento (a cidade), com poucos pixels inválidos e pelo menos
alagamento_min km² de alagamento (pelo SOM, no indice.csv), a que tem a mistura
mais equilibrada de água e terra (fração de água perto de 50%), preferindo as
mais próximas do centro: áreas quase todas cobertas de água mostram pouco. Lê os dados brutos da área, aplica os
três métodos com as mesmas três classes (não alagamento, água permanente,
alagamento) e faz uma figura com uma linha por evento: imagem de antes, imagem
da enchente, SOM, NDWI e MNDWI.

Uso:
    python figuraMetodos.py                      # gera resultados/comparacao_metodos.png
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import enchenteSOM
from enchenteSOM import AREA_PIXEL_KM2, CLASSES, CORES_CLASSES


dist_max = 2          # Distância máxima, em áreas, do centro do evento
invalido_max = 10     # % máxima de pixels inválidos na área escolhida
alagamento_min = 2    # Alagamento mínimo (km²) da área escolhida
penalidade_dist = 0.03  # Desconto na nota por área de distância do centro
limiar_agua = 0       # Índice acima disso indica água (NDWI e MNDWI)

# Nome de cada pasta na figura
NOMES = {"porto_alegre": "Porto Alegre (BR, 2024)", "pelotas": "Pelotas (BR, 2024)",
         "restinga_seca": "Restinga Sêca (BR, 2024)", "kaoh_chen": "Kaoh Chen (Camboja, 2017)",
         "sa_khut": "Sa Khut (Tailândia, 2017)", "phrai_bueng": "Phrai Bueng (Tailândia, 2017)",
         "assuncao": "Assunção (Paraguai, 2016)", "dadu": "Dadu (Paquistão, 2022)",
         "conselice": "Conselice (Itália, 2023)", "nilo_branco": "Nilo Branco (Sudão, 2019)",
         "palamas": "Palamas (Grécia, 2023)", "uralsk": "Uralsk (Cazaquistão, 2024)",
         "lokoja": "Lokoja (Nigéria, 2022)"}


# Nota de uma área: p * (1 - p), onde p é a fração de água (alagamento + permanente),
# menos a penalidade pela distância do centro
def nota(d):
    e = d["enchente"]
    p = (float(e["som_alagamento_km2"]) + float(e["som_agua_permanente_km2"])) / (
        int(e["altura"]) * int(e["largura"]) * AREA_PIXEL_KM2)
    dist = max(abs(int(e["linha_grade"])), abs(int(e["coluna_grade"])))
    return p * (1 - p) - penalidade_dist * dist

# Escolhe a área com a mistura mais equilibrada de água e terra perto do centro do evento
def escolher_area(pasta):
    with open(os.path.join(pasta, "indice.csv"), encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    por_area = {}
    for l in linhas:
        por_area.setdefault(l["area"], {})[l["estado"]] = l
    candidatas = [d for d in por_area.values()
                  if max(abs(int(d["enchente"]["linha_grade"])), abs(int(d["enchente"]["coluna_grade"]))) <= dist_max
                  and float(d["enchente"]["pct_invalido"]) <= invalido_max
                  and float(d["enchente"]["som_alagamento_km2"]) >= alagamento_min]
    if not candidatas:  # Nenhuma área perto do centro atende: considera todas as da pasta
        candidatas = list(por_area.values())
    return max(candidatas, key=nota)

def ler_bruto(arquivo):
    with rasterio.open(arquivo) as src:
        return src.read().transpose(1, 2, 0).astype(float) / 10000

def ler_mascara(arquivo):
    with rasterio.open(arquivo) as src:
        return src.read(1).astype(bool)

# Composição RGB com contraste por percentis (mesma escala nas duas datas)
def rgb_auto(refl_antes, refl_depois, invalidos):
    todos = np.concatenate([r[:, :, [2, 1, 0]][~invalidos] for r in (refl_antes, refl_depois)])
    lo, hi = np.percentile(todos, [2, 98]) if len(todos) else (0, 0.15)
    return [np.clip((r[:, :, [2, 1, 0]] - lo) / max(hi - lo, 1e-6), 0, 1) for r in (refl_antes, refl_depois)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara SOM, NDWI e MNDWI em uma área de cada evento")
    parser.add_argument("--pasta-dataset", default="dataset")
    parser.add_argument("--saida", default=os.path.join("resultados", "comparacao_metodos.png"))
    args = parser.parse_args()

    pastas = [p for p in sorted(glob.glob(os.path.join(args.pasta_dataset, "*")))
              if os.path.exists(os.path.join(p, "indice.csv"))]
    ordem = list(NOMES)
    pastas.sort(key=lambda p: ordem.index(os.path.basename(p)) if os.path.basename(p) in ordem else len(ordem))

    resultados = []
    for pasta in pastas:
        evento = os.path.basename(pasta)
        d = escolher_area(pasta)
        refl_antes = ler_bruto(os.path.join(pasta, d["antes"]["arquivo_bruto"]))
        refl_depois = ler_bruto(os.path.join(pasta, d["enchente"]["arquivo_bruto"]))
        invalidos = ler_mascara(os.path.join(pasta, d["enchente"]["arquivo_mascara"]))

        # SOM bi-temporal (unidades classificadas pelo MNDWI)
        entrada, _ = enchenteSOM.montar_entrada(refl_antes, refl_depois)
        SOM, result = enchenteSOM.treinar_e_segmentar(entrada, invalidos)
        classe_unidade, _ = enchenteSOM.classificar_unidades(result, invalidos, refl_antes, refl_depois, enchenteSOM.SOM_SIZE, limiar_agua)
        mapas = {"SOM": classe_unidade[result],
                 "NDWI": enchenteSOM.mapa_indice(refl_antes, refl_depois, limiar_agua, "NDWI"),
                 "MNDWI": enchenteSOM.mapa_indice(refl_antes, refl_depois, limiar_agua, "MNDWI")}
        areas = {m: enchenteSOM.areas_por_classe(mapa, invalidos) for m, mapa in mapas.items()}
        resultados.append({"evento": evento, "dados": d, "mapas": mapas, "areas": areas, "invalidos": invalidos,
                           "rgb": rgb_auto(refl_antes, refl_depois, invalidos)})
        print("%-14s %s | alagamento SOM %.2f, NDWI %.2f, MNDWI %.2f km²"
              % (evento, d["enchente"]["area"], areas["SOM"][2], areas["NDWI"][2], areas["MNDWI"][2]), flush=True)

    # Figura: uma linha por evento
    n = len(resultados)
    altura = 4.4 * n + 1.6
    fig, ax = plt.subplots(n, 5, figsize=(20, altura), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(left=0.075, right=0.995, top=1 - 1.1 / altura, bottom=0.6 / altura, wspace=0.04, hspace=0.22)
    cmap = ListedColormap(CORES_CLASSES)
    for k, r in enumerate(resultados):
        d = r["dados"]
        ax[k, 0].imshow(r["rgb"][0])
        ax[k, 0].set_title("Antes\n" + "/".join(reversed(d["antes"]["data"].split("-"))), fontsize=10.5)
        ax[k, 1].imshow(r["rgb"][1])
        ax[k, 1].set_title("Enchente\n" + "/".join(reversed(d["enchente"]["data"].split("-"))), fontsize=10.5)
        for m, nome in enumerate(["SOM", "NDWI", "MNDWI"]):
            mapa = r["mapas"][nome]
            a = ax[k, m + 2]
            a.imshow(r["rgb"][1])
            a.imshow(np.ma.masked_where(mapa == 0, mapa), cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, alpha=0.85)
            a.imshow(np.ma.masked_where(~r["invalidos"], np.ones(mapa.shape)), cmap=ListedColormap(["white"]), alpha=0.9)
            a.set_title("%s\nalagamento %.1f km², permanente %.1f km²"
                        % ("SOM bi-temporal" if nome == "SOM" else nome + " > 0", r["areas"][nome][2],
                           r["areas"][nome][1]), fontsize=10.5)
        ax[k, 0].set_ylabel("%s\n%s\nlat %s, lon %s" % (NOMES.get(r["evento"], r["evento"]), d["enchente"]["area"],
                                                       d["enchente"]["lat_centro"], d["enchente"]["lon_centro"]),
                            fontsize=11)
    legenda = [Patch(color=CORES_CLASSES[c], label=CLASSES[c]) for c in (1, 2)]
    legenda.append(Patch(facecolor="white", edgecolor="k", label="Nuvem ou sem dado"))
    fig.legend(handles=legenda, loc="lower center", ncol=3, fontsize=12, frameon=False)
    fig.suptitle("SOM bi-temporal x limiar direto de NDWI e MNDWI - uma área (~5,5 km) de cada evento, perto da cidade",
                 fontsize=15, y=1 - 0.25 / altura)
    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    fig.savefig(args.saida, dpi=70)

    resumo = [{"evento": r["evento"], "area": r["dados"]["enchente"]["area"],
               "lat": r["dados"]["enchente"]["lat_centro"], "lon": r["dados"]["enchente"]["lon_centro"],
               **{"%s_%s_km2" % (m.lower(), c): round(float(r["areas"][m][i]), 3)
                  for m in r["areas"] for i, c in ((1, "permanente"), (2, "alagamento"))}} for r in resultados]
    with open(os.path.splitext(args.saida)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1)
    print("Figura salva em", args.saida)
