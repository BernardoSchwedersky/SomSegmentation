# -*- coding: utf-8 -*-
"""
Figura com exemplos do radar Sentinel-1 na detecção do alagamento
(experimentoRadar.py): para cada recorte, imagens de antes e do evento, o VH do
evento, a referência e os mapas do MNDWI e do SOM óptico pré-treinado, sem e com
o radar na regra (VH < -22 dB onde o MNDWI do evento > -0,2).

Uso:
    python figuraExemplosRadar.py <pasta_cache> resultados/sen1floods11/radar_exemplos.png [recortes...]
"""

import glob
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import enchenteSOM as es
import experimentoRadar as er
from avaliarSen1Floods11 import contar, metricas
from experimentoAreaAlagada import ler_jrc
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from variacaoUnidadesSOM import ler_recorte

LIMIAR, MINIMO = -22, -0.2

C, saida = sys.argv[1], sys.argv[2]
casos = sys.argv[3:] or ["Paraguay_62897", "Paraguay_305760", "Somalia_322855"]
por_evento = defaultdict(list)
for f in glob.glob(C + "/resultados/*.json"):
    r = json.load(open(f))
    if r["status"] == "ok":
        por_evento[r["pais"]].append(r)
imagens = sorted(e["recorte"] for e in (escolher_pre_treino(l) for l in por_evento.values())
                 if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX)
SOM = er.pre_treinar_radar(imagens, False, 0, C)

cores_classe = ListedColormap(["#08306b", "#3fa9f5"])
fig, ax = plt.subplots(len(casos), 8, figsize=(32, 4.4 * len(casos)), subplot_kw=dict(xticks=[], yticks=[]))
ax = np.atleast_2d(ax)
for k, rec in enumerate(casos):
    r = json.load(open(C + "/resultados/" + rec + ".json"))
    rotulo, antes, depois, inv = ler_recorte(r, C)
    vv, vh = er.ler_radar(rec, C)
    inv = inv | np.isnan(vh)
    jrc = ler_jrc(rec, C)
    validos = (rotulo >= 0) & ~inv
    ref = np.zeros(rotulo.shape, int)
    ref[(rotulo == 1) & jrc] = 1
    ref[(rotulo == 1) & ~jrc] = 2
    result = es.segmentar(SOM, er.montar(antes, depois, vv, vh, False), 1)
    mapas = [("MNDWI", er.mapa_pixel(antes, depois, vh)),
             ("MNDWI + radar", er.mapa_pixel(antes, depois, vh, LIMIAR, MINIMO)),
             ("SOM óptico", er.classificar(result, inv, antes, depois, vh, es.SOM_SIZE)[result]),
             ("SOM óptico + radar", er.classificar(result, inv, antes, depois, vh, es.SOM_SIZE, LIMIAR, MINIMO)[result])]
    lo, hi = np.percentile(np.concatenate([x[:, :, [2, 1, 0]][validos] for x in (antes, depois)]), [2, 98])
    rgb = [np.clip((x[:, :, [2, 1, 0]] - lo) / (hi - lo), 0, 1) for x in (antes, depois)]
    ax[k, 0].imshow(rgb[0]); ax[k, 0].set_title("Antes %s" % r["data_antes"], fontsize=12)
    ax[k, 1].imshow(rgb[1]); ax[k, 1].set_title("Evento %s" % r["data_depois"], fontsize=12)
    ax[k, 2].imshow(vh, cmap="gray", vmin=-30, vmax=-8); ax[k, 2].set_title("Radar VH do evento (dB)", fontsize=12)

    def classes(a, mapa, titulo):
        a.imshow(rgb[1])
        m = np.where(validos, mapa, 0)
        a.imshow(np.ma.masked_where(m == 0, m), cmap=cores_classe, vmin=0.5, vmax=2.5, alpha=0.85)
        a.imshow(np.ma.masked_where(validos, np.ones(m.shape)), cmap=ListedColormap(["white"]), alpha=0.9)
        a.set_title(titulo, fontsize=12)
    classes(ax[k, 3], ref, "Referência")
    for col, (nome, mapa) in enumerate(mapas, start=4):
        classes(ax[k, col], mapa, "%s - IoU %.2f" % (nome, metricas(contar(mapa == 2, ref == 2, validos))["IoU"]))
    ax[k, 0].set_ylabel("%s\nalagamento %.0f%%" % (rec, 100 * (ref == 2)[validos].mean()), fontsize=12)
leg = [Patch(color="#08306b", label="Água permanente"), Patch(color="#3fa9f5", label="Alagamento"),
       Patch(facecolor="white", edgecolor="k", label="Sem dado / nuvem")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=14, frameon=False)
fig.suptitle("Radar Sentinel-1 do evento na regra do alagamento (água no evento = MNDWI > 0 ou VH < %d dB onde MNDWI > %g)"
             % (LIMIAR, MINIMO), fontsize=16)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(saida, dpi=60)
