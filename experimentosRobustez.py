# -*- coding: utf-8 -*-
"""
Experimentos controlados para identificar onde o SOM bi-temporal é mais robusto
que os limiares diretos de NDWI e MNDWI.

Áreas de teste: áreas do dataset em que os três métodos concordam na imagem
limpa. A referência são os pixels em que os três métodos concordam sobre o
alagamento na imagem limpa (os demais pixels não entram na avaliação). Em
seguida a imagem é degradada de forma controlada e cada método é avaliado
contra essa referência pela F1 da classe alagamento:

1. Ruído: ruído gaussiano somado à reflectância das duas datas (sigma).
2. Névoa: deslocamento aditivo na imagem da enchente, maior no azul e no verde
   (como névoa ou nuvem fina), o que aumenta artificialmente os índices de água.
3. Desbalanceamento entre bandas: o infravermelho próximo (B08) da imagem da
   enchente multiplicado por um fator (calibração ou correção atmosférica
   diferente entre as bandas). Um ganho igual em todas as bandas não muda os
   índices normalizados, por isso só uma banda é alterada.
4. Limiar: área de alagamento do NDWI e do MNDWI para limiares de -0,2 a 0,2,
   comparada com o SOM, que não usa limiar fixo nos pixels.

Lê os dados brutos de dataset/ (gerados por criarDataset.py).

Uso:
    python experimentosRobustez.py      # gera resultados/robustez/
"""

import argparse
import csv
import json
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

import enchenteSOM
from enchenteSOM import CORES_CLASSES

# Áreas de teste (pasta do dataset, área): concordância alta entre os métodos na imagem limpa
AREAS = [("dadu", "N00_O02"), ("uralsk", "S01_L01"), ("conselice", "N00_L00"),
         ("palamas", "N00_L00"), ("kaoh_chen", "N00_L00")]

# Níveis de cada degradação (o primeiro é a imagem limpa)
RUIDO = [0, 0.01, 0.02, 0.04, 0.06]
NEVOA = [0, 0.02, 0.04, 0.06, 0.08]
GANHO = [1.0, 0.9, 0.8, 0.7, 0.6]
LIMIARES = [-0.2, -0.1, 0, 0.1, 0.2]

# Peso da névoa em cada banda (B02, B03, B04, B08, B11): maior no azul, quase nada no SWIR
PESO_NEVOA = np.array([1.0, 0.8, 0.6, 0.3, 0.1])

METODOS = ["SOM", "NDWI", "MNDWI"]


def ler_area(pasta, area):
    with open(os.path.join("dataset", pasta, "indice.csv"), encoding="utf-8") as f:
        linhas = {l["estado"]: l for l in csv.DictReader(f) if l["area"] == area}
    def bruto(l):
        with rasterio.open(os.path.join("dataset", pasta, l["arquivo_bruto"])) as src:
            return src.read().transpose(1, 2, 0).astype(float) / 10000
    with rasterio.open(os.path.join("dataset", pasta, linhas["enchente"]["arquivo_mascara"])) as src:
        invalidos = src.read(1).astype(bool)
    return bruto(linhas["antes"]), bruto(linhas["enchente"]), invalidos

def metodos(refl_antes, refl_depois, invalidos, seed=0):
    entrada, _ = enchenteSOM.montar_entrada(refl_antes, refl_depois)
    _, result = enchenteSOM.treinar_e_segmentar(entrada, invalidos, seed=seed)
    classe_unidade, _ = enchenteSOM.classificar_unidades(result, invalidos, refl_antes, refl_depois, 12)
    return {"SOM": classe_unidade[result],
            "NDWI": enchenteSOM.mapa_indice(refl_antes, refl_depois, 0, "NDWI"),
            "MNDWI": enchenteSOM.mapa_indice(refl_antes, refl_depois, 0, "MNDWI")}

def f1(previsto, referencia, avaliar):
    p, r = previsto[avaliar], referencia[avaliar]
    vp = np.sum(p & r)
    return 2 * vp / max(np.sum(p) + np.sum(r), 1)

def degradar(refl_antes, refl_depois, tipo, nivel, rng):
    a, d = refl_antes.copy(), refl_depois.copy()
    if tipo == "ruido":
        a += rng.normal(0, nivel, a.shape)
        d += rng.normal(0, nivel, d.shape)
    elif tipo == "nevoa":
        d += nivel * PESO_NEVOA
    elif tipo == "ganho":
        d[:, :, enchenteSOM.BANDAS.index("B08")] *= nivel
    return np.clip(a, 0, 1), np.clip(d, 0, 1)

def rodar(args):
    pasta, area, tipo, nivel = args
    refl_antes, refl_depois, invalidos = ler_area(pasta, area)
    a, d = degradar(refl_antes, refl_depois, tipo, nivel, np.random.RandomState(1))
    mapas = metodos(a, d, invalidos)
    return (pasta, area, tipo, nivel), {m: (mapa == 2) for m, mapa in mapas.items()}, mapas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa a robustez do SOM, NDWI e MNDWI a degradações da imagem")
    parser.add_argument("--saida", default=os.path.join("resultados", "robustez"))
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    tarefas = [(p, a, "limpa", 0) for p, a in AREAS]
    tarefas += [(p, a, t, n) for p, a in AREAS for t, niveis in (("ruido", RUIDO), ("nevoa", NEVOA), ("ganho", GANHO))
                for n in niveis[1:]]
    with Pool(args.processos) as pool:
        saidas = pool.map(rodar, tarefas)
    alag = {k: v for k, v, _ in saidas}
    mapas = {k: m for k, _, m in saidas}

    # Referência: pixels em que os três métodos concordam na imagem limpa
    referencia = {}
    for p, a in AREAS:
        limpa = alag[(p, a, "limpa", 0)]
        _, _, invalidos = ler_area(p, a)
        concordam = (limpa["SOM"] == limpa["NDWI"]) & (limpa["NDWI"] == limpa["MNDWI"]) & ~invalidos
        referencia[(p, a)] = (limpa["SOM"], concordam)
        print("%-10s %s: %.0f%% dos pixels válidos com concordância, %.1f%% alagados"
              % (p, a, 100 * concordam.sum() / (~invalidos).sum(), 100 * limpa["SOM"][concordam].mean()))

    resultados = {}
    for tipo, niveis in (("ruido", RUIDO), ("nevoa", NEVOA), ("ganho", GANHO)):
        resultados[tipo] = {m: [] for m in METODOS}
        for n in niveis:
            for m in METODOS:
                vals = []
                for p, a in AREAS:
                    ref, avaliar = referencia[(p, a)]
                    chave = (p, a, "limpa", 0) if n == niveis[0] else (p, a, tipo, n)
                    vals.append(f1(alag[chave][m], ref, avaliar))
                resultados[tipo][m].append(vals)
        print("\n%s" % tipo)
        for m in METODOS:
            print("  %-6s " % m + "  ".join("%.3f" % np.mean(v) for v in resultados[tipo][m]))

    # Sensibilidade ao limiar: área de alagamento (km²) em função do limiar
    limiar = {}
    for p, a in AREAS:
        refl_antes, refl_depois, invalidos = ler_area(p, a)
        limiar[p] = {ind: [float(enchenteSOM.areas_por_classe(
            enchenteSOM.mapa_indice(refl_antes, refl_depois, t, ind), invalidos)[2]) for t in LIMIARES]
            for ind in ("NDWI", "MNDWI")}
        limiar[p]["SOM"] = float(enchenteSOM.areas_por_classe(mapas[(p, a, "limpa", 0)]["SOM"], invalidos)[2])

    with open(os.path.join(args.saida, "robustez.json"), "w", encoding="utf-8") as f:
        json.dump({"areas": AREAS, "ruido": RUIDO, "nevoa": NEVOA, "ganho": GANHO, "limiares": LIMIARES,
                   "f1": {t: {m: [[round(x, 4) for x in v] for v in vals] for m, vals in r.items()}
                          for t, r in resultados.items()},
                   "area_por_limiar_km2": limiar}, f, ensure_ascii=False, indent=1)

    # Figura 1: F1 em função de cada degradação
    cores = {"SOM": "#3fa9f5", "NDWI": "#e7298a", "MNDWI": "#ff9f1c"}
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    for k, (tipo, niveis, rotulo) in enumerate((("ruido", RUIDO, "Ruído gaussiano (sigma da reflectância)"),
                                                ("nevoa", NEVOA, "Névoa na imagem da enchente (reflectância no azul)"),
                                                ("ganho", GANHO, "Ganho no infravermelho próximo (B08) da enchente"))):
        ordem = np.argsort(niveis)
        x = np.array(niveis)[ordem]
        for m in METODOS:
            v = np.array(resultados[tipo][m])[ordem]
            ax[k].plot(x, v.mean(axis=1), "o-", color=cores[m], lw=2.5, label=m)
            ax[k].fill_between(x, v.min(axis=1), v.max(axis=1), color=cores[m], alpha=0.12)
        ax[k].set_xlabel(rotulo)
        ax[k].set_ylabel("F1 do alagamento (referência: imagem limpa)")
        ax[k].set_ylim(0, 1.02)
        ax[k].grid(alpha=0.3)
        ax[k].legend()
    for p, a in AREAS:
        r = limiar[p]
        l, = ax[3].plot(LIMIARES, np.array(r["NDWI"]) / max(r["SOM"], 1e-6), "o-", lw=2, label=p)
        ax[3].plot(LIMIARES, np.array(r["MNDWI"]) / max(r["SOM"], 1e-6), "s--", lw=1.2, color=l.get_color())
    ax[3].axhline(1, color="k", lw=1)
    ax[3].set_xlabel("Limiar do índice (NDWI: linha cheia, MNDWI: tracejada)")
    ax[3].set_ylabel("Área de alagamento / área do SOM")
    ax[3].grid(alpha=0.3)
    ax[3].legend(fontsize=9)
    ax[3].set_title("Sensibilidade ao limiar")
    ax[0].set_title("Ruído")
    ax[1].set_title("Névoa / nuvem fina")
    ax[2].set_title("Desbalanceamento entre bandas")
    fig.suptitle("Robustez dos métodos a degradações controladas (média de %d áreas; faixa = mínimo e máximo)"
                 % len(AREAS), fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "robustez_curvas.png"), dpi=80)

    # Figura 2: exemplos de mapas no nível mais forte de cada degradação
    exemplos = [("limpa", 0), ("ruido", RUIDO[-1]), ("nevoa", NEVOA[-1]), ("ganho", GANHO[-1])]
    p, a = AREAS[2]
    refl_antes, refl_depois, invalidos = ler_area(p, a)
    fig, ax = plt.subplots(len(exemplos), 4, figsize=(16, 4.1 * len(exemplos)), subplot_kw=dict(xticks=[], yticks=[]))
    ref, avaliar = referencia[(p, a)]
    for k, (tipo, n) in enumerate(exemplos):
        _, d = degradar(refl_antes, refl_depois, tipo, n, np.random.RandomState(1)) if tipo != "limpa" else (None, refl_depois)
        lo, hi = np.percentile(refl_depois[:, :, [2, 1, 0]][~invalidos], [2, 98])
        rgb = np.clip((d[:, :, [2, 1, 0]] - lo) / (hi - lo), 0, 1)
        ax[k, 0].imshow(rgb)
        ax[k, 0].set_ylabel({"limpa": "Imagem limpa", "ruido": "Ruído sigma %.2f" % n,
                             "nevoa": "Névoa %.2f" % n, "ganho": "Ganho no B08 %.2f" % n}[tipo], fontsize=12)
        chave = (p, a, tipo, n)
        for m, metodo in enumerate(METODOS):
            mapa = mapas[chave][metodo]
            ax[k, m + 1].imshow(rgb)
            ax[k, m + 1].imshow(np.ma.masked_where(mapa == 0, mapa), cmap=ListedColormap(CORES_CLASSES),
                                vmin=-0.5, vmax=2.5, alpha=0.85)
            ax[k, m + 1].set_title("%s - F1 %.2f" % (metodo, f1(alag[chave][metodo], ref, avaliar)), fontsize=11)
    ax[0, 0].set_title("Enchente (%s %s)" % (p, a), fontsize=11)
    fig.suptitle("Exemplos de degradação: %s %s" % (p, a), fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "robustez_exemplos.png"), dpi=70)
    print("\nFiguras salvas em", args.saida)
