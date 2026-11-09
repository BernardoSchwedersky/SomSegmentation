# -*- coding: utf-8 -*-
"""
Figura com exemplos da avaliação só da área alagada (experimentoAreaAlagada.py):
para cada recorte, imagens de antes e do evento, a referência (rótulo à mão sem a
água permanente do JRC), os mapas do MNDWI e do SOM pré-treinado (uma imagem de
cada evento, normalização fixa) e os erros do alagamento do SOM.

Uso:
    python figuraExemplosAlagada.py <pasta_cache> resultados/sen1floods11/area_alagada_exemplos.png
"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import enchenteSOM as es
from avaliarSen1Floods11 import contar, metricas
from variacaoUnidadesSOM import ler_recorte
from experimentoAreaAlagada import ler_jrc
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoSOM import pre_treinar
import glob
from collections import defaultdict

C, saida = sys.argv[1], sys.argv[2]
casos = [("Somalia_322855", "Acerto típico (Somália)"), ("USA_86502", "Lago com faixa alagada (EUA)"),
         ("Paraguay_305760", "SOM melhor que o MNDWI (Paraguai)"),
         ("Nigeria_81933", "Imagem de antes já alagada (Nigéria)"),
         ("Paraguay_62897", "Vegetação alagada não vista (Paraguai)")]
por_evento = defaultdict(list)
for f in glob.glob(C + "/resultados/*.json"):
    r = json.load(open(f))
    if r["status"] == "ok":
        por_evento[r["pais"]].append(r)
imagens = sorted(e["recorte"] for e in (escolher_pre_treino(l) for l in por_evento.values())
                 if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX)
SOM, _ = pre_treinar(imagens, "fixa", 0, C)

cores_classe = ListedColormap(["#08306b", "#3fa9f5"])  # 1 permanente, 2 alagamento
cores_erro = ListedColormap(["#3fa9f5", "#d95f0e", "#ffd92f"])  # acerto, falso positivo, falso negativo
fig, ax = plt.subplots(len(casos), 6, figsize=(24, 4.3 * len(casos)), subplot_kw=dict(xticks=[], yticks=[]))
for k, (rec, titulo) in enumerate(casos):
    r = json.load(open(C + "/resultados/" + rec + ".json"))
    rotulo, antes, depois, inv = ler_recorte(r, C)
    jrc = ler_jrc(rec, C)
    validos = (rotulo >= 0) & ~inv
    ref = np.zeros(rotulo.shape, int)
    ref[(rotulo == 1) & jrc] = 1
    ref[(rotulo == 1) & ~jrc] = 2
    mapa_mndwi = es.mapa_indice(antes, depois, 0, "MNDWI")
    entrada, _ = es.montar_entrada(antes, depois, normalizacao="fixa")
    result = es.segmentar(SOM, entrada, 1)
    classe, _ = es.classificar_unidades(result, inv, antes, depois, es.SOM_SIZE)
    mapa_som = classe[result]
    lo, hi = np.percentile(np.concatenate([x[:, :, [2, 1, 0]][validos] for x in (antes, depois)]), [2, 98])
    rgb = [np.clip((x[:, :, [2, 1, 0]] - lo) / (hi - lo), 0, 1) for x in (antes, depois)]
    ax[k, 0].imshow(rgb[0]); ax[k, 0].set_title("Antes %s" % r["data_antes"], fontsize=11)
    ax[k, 1].imshow(rgb[1]); ax[k, 1].set_title("Evento %s" % r["data_depois"], fontsize=11)

    def classes(a, mapa, nome):
        a.imshow(rgb[1])
        m = np.where(validos, mapa, 0)
        a.imshow(np.ma.masked_where(m == 0, m), cmap=cores_classe, vmin=0.5, vmax=2.5, alpha=0.85)
        a.imshow(np.ma.masked_where(validos, np.ones(m.shape)), cmap=ListedColormap(["white"]), alpha=0.9)
        a.set_title(nome, fontsize=11)
    classes(ax[k, 2], ref, "Referência (rótulo + JRC)")
    for col, (mapa, nome) in ((3, (mapa_mndwi, "MNDWI")), (4, (mapa_som, "SOM pré-treinado"))):
        iou = metricas(contar(mapa == 2, ref == 2, validos))["IoU"]
        classes(ax[k, col], mapa, "%s - IoU do alagamento %.2f" % (nome, iou))
    # Erros do alagamento: MNDWI e SOM lado a lado na mesma coluna seria confuso; mostra os do SOM
    erro = np.full(ref.shape, -1)
    p, g = (mapa_som == 2) & validos, (ref == 2) & validos
    erro[p & g] = 0; erro[p & ~g] = 1; erro[~p & g] = 2
    ax[k, 5].imshow(rgb[1] * 0.45 + 0.55)
    ax[k, 5].imshow(np.ma.masked_where(erro < 0, erro), cmap=cores_erro, vmin=-0.5, vmax=2.5)
    ax[k, 5].set_title("Erros do alagamento (SOM)", fontsize=11)
    ax[k, 0].set_ylabel("%s\n%s\nalagamento %.0f%%" % (titulo, rec, 100 * g.sum() / validos.sum()), fontsize=11)
leg = [Patch(color="#08306b", label="Água permanente"), Patch(color="#3fa9f5", label="Alagamento"),
       Patch(facecolor="white", edgecolor="k", label="Sem dado / nuvem"),
       Patch(color="#d95f0e", label="Falso alagamento"), Patch(color="#ffd92f", label="Alagamento perdido")]
fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=13, frameon=False)
fig.suptitle("Só a área alagada: exemplos do Sen1Floods11 (referência = rótulo à mão sem a água permanente do JRC)",
             fontsize=16)
fig.tight_layout(rect=(0, 0.025, 1, 0.975))
fig.savefig(saida, dpi=60)
