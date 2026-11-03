# -*- coding: utf-8 -*-
"""
Testa o pré-treino do SOM bi-temporal em uma imagem balanceada entre água e
terra e a aplicação desse mesmo SOM, sem alterá-lo, nos outros recortes do
Sen1Floods11.

Pré-treino: recortes com ~50% de água no rótulo (padrão: India_399883,
Mekong_922373 e USA_788696, separados e também juntos), com o padrão de
enchenteSOM (36 unidades, 2% dos pixels). Cada SOM pré-treinado segmenta os
demais recortes (os de pré-treino ficam fora da avaliação) e as unidades são
classificadas de duas formas:
- "classe fixa": a classe que cada unidade recebeu na imagem de pré-treino;
- "classe local": a classe recalculada em cada recorte pelo MNDWI médio dos
  pixels da unidade (como no método atual).
Com duas normalizações da entrada: "percentil" (atual, cada imagem pela sua
escala) e "fixa" (mesma escala para todas as imagens).

A referência é o SOM treinado em cada recorte (resultado de
experimentoFeaturesCNN.py, mesma pasta de cache, 36 unidades) e os índices.

Uso:
    python experimentoPreTreinoSOM.py --sementes 3
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import enchenteSOM
from avaliarSen1Floods11 import contar, metricas, somar
from variacaoUnidadesSOM import ler_recorte

PRE_TREINO = ["India_399883", "Mekong_922373", "USA_788696"]
NORMALIZACOES = ["percentil", "fixa"]


# Treina um SOM em um ou mais recortes (as amostras de cada recorte são juntadas)
# e guarda a classe de cada unidade na(s) imagem(ns) de pré-treino
def pre_treinar(recortes, normalizacao, seed, pasta_cache):
    dados, amostras = [], []
    rng = np.random.RandomState(seed)
    for r in recortes:
        rotulo, antes, depois, invalidos = ler_recorte({"recorte": r}, pasta_cache)
        entrada, _ = enchenteSOM.montar_entrada(antes, depois, normalizacao=normalizacao)
        validos = np.flatnonzero(~invalidos.ravel())
        sel = rng.choice(validos, int(round(len(validos) * 0.02)), replace=False)
        amostras.append(enchenteSOM.extrair_janelas(entrada, 1, sel))
        dados.append((entrada, antes, depois, invalidos))
    treino = np.concatenate(amostras)
    rng.shuffle(treino)
    enchenteSOM.somSeg.rand.seed(seed)
    SOM = enchenteSOM.somSeg.create_map(enchenteSOM.SOM_SIZE, 1, treino.shape[1])
    SOM = enchenteSOM.somSeg.train_SOM(SOM, treino)
    # Classe de cada unidade: MNDWI médio dos pixels das imagens de pré-treino
    results = [enchenteSOM.segmentar(SOM, e, 1) for e, _, _, _ in dados]
    juntar = lambda lista: np.concatenate([x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.ravel() for x in lista])
    res_j = juntar(results)[:, None]
    classe, _ = enchenteSOM.classificar_unidades(res_j, juntar([d[3] for d in dados])[:, None],
                                                juntar([d[1] for d in dados])[:, None, :],
                                                juntar([d[2] for d in dados])[:, None, :], enchenteSOM.SOM_SIZE)
    return SOM, classe

def avaliar(args):
    r, modelos, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "pre_treino", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    validos = (rotulo >= 0) & ~invalidos
    referencia = rotulo == 1
    entradas = {n: enchenteSOM.montar_entrada(antes, depois, normalizacao=n)[0] for n in NORMALIZACOES}
    res = {"recorte": r["recorte"], "pct_agua_rotulo": r["pct_agua_rotulo"], "variacoes": {}}
    for (nome_pre, norm), lista in modelos.items():
        for SOM, classe_fixa in lista:
            result = enchenteSOM.segmentar(SOM, entradas[norm], 1)
            classe_local, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, enchenteSOM.SOM_SIZE)
            for tipo, classe in (("classe fixa", classe_fixa), ("classe local", classe_local)):
                chave = "%s | %s | %s" % (nome_pre, norm, tipo)
                res["variacoes"].setdefault(chave, []).append(
                    contar(np.isin(classe[result], [1, 2]), referencia, validos))
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s água %5.1f%%" % (r["recorte"], r["pct_agua_rotulo"]), flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o SOM pré-treinado em uma imagem balanceada")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.pasta_cache, "pre_treino"), exist_ok=True)

    # SOMs pré-treinados: cada imagem separada e as três juntas, nas duas normalizações
    grupos = {r: [r] for r in PRE_TREINO}
    grupos["3 imagens"] = PRE_TREINO
    modelos = {}
    for nome, recortes in grupos.items():
        for norm in NORMALIZACOES:
            modelos[(nome, norm)] = [pre_treinar(recortes, norm, s, args.pasta_cache) for s in range(args.sementes)]
            print("Pré-treino %-14s %-9s: unidades de água por semente %s" % (nome, norm, [int(np.isin(c, [1, 2]).sum())
                  for _, c in modelos[(nome, norm)]]), flush=True)

    recortes = []
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok" and r["recorte"] not in PRE_TREINO:
            recortes.append(r)
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar, [(r, modelos, args.pasta_cache) for r in recortes], chunksize=1)

    # Referência: SOM treinado em cada recorte e índices (experimentoFeaturesCNN.py)
    referencia = {}
    for r in resultados:
        with open(os.path.join(args.pasta_cache, "features_cnn", r["recorte"] + ".json")) as f:
            referencia[r["recorte"]] = json.load(f)
    faixas = [("Todos", 0, 101), ("0-1%", 0, 1), ("1-5%", 1, 5), ("5-15%", 5, 15), ("15-40%", 15, 40), ("40-100%", 40, 101)]
    variacoes = list(resultados[0]["variacoes"])

    def contagens(sel, v, s):
        if v == "SOM por recorte":
            return [referencia[r["recorte"]]["SOM"]["SOM"][s] for r in sel]
        if v in ("NDWI", "MNDWI"):
            return [referencia[r["recorte"]][v] for r in sel]
        return [r["variacoes"][v][s] for r in sel]

    nomes = ["SOM por recorte"] + variacoes + ["NDWI", "MNDWI"]
    tabela = {}
    for nome_faixa, a, b in faixas:
        sel = [r for r in resultados if a <= r["pct_agua_rotulo"] < b]
        tabela[nome_faixa] = {"n": len(sel)}
        for n in nomes:
            ms = [metricas(somar(contagens(sel, n, s))) for s in range(args.sementes)]
            tabela[nome_faixa][n] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            tabela[nome_faixa][n]["IoU_min_semente"] = float(min(m["IoU"] for m in ms))
    com_agua = [r for r in resultados if r["pct_agua_rotulo"] >= 1]
    falhas = {n: sum(1 for r in com_agua if np.mean([metricas(c)["IoU"] for c in contagens([r], n, 0)]
                                                     if n in ("NDWI", "MNDWI") else
                                                     [metricas(contagens([r], n, s)[0])["IoU"] for s in range(args.sementes)])
                     < 0.05) for n in nomes}

    print("\n%d recortes avaliados (sem os de pré-treino)" % len(resultados))
    print("%-48s %6s %6s %6s %6s %6s %6s %7s" % ("Variação", "Todos", "0-1%", "1-5%", "5-15%", "15-40%", ">40%", "falhas"))
    for n in nomes:
        print("%-48s " % n + " ".join("%6.3f" % tabela[f][n]["IoU"] for f, _, _ in faixas) + " %7d" % falhas[n])

    with open(os.path.join(args.saida, "pre_treino_som.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "pre_treino": PRE_TREINO, "sementes": args.sementes,
                   "unidades": enchenteSOM.SOM_SIZE, "metricas": tabela, "falhas_recortes_com_agua": falhas,
                   "recortes_com_agua": len(com_agua)}, f, ensure_ascii=False, indent=1)

    # Figura: IoU de cada variação (todos os recortes e por faixa de água)
    fig, ax = plt.subplots(1, 2, figsize=(22, 7.5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    y = np.arange(len(nomes))[::-1]
    cores = []
    for n in nomes:
        if n == "SOM por recorte":
            cores.append("#3fa9f5")
        elif n == "NDWI":
            cores.append("#e7298a")
        elif n == "MNDWI":
            cores.append("#ff9f1c")
        else:
            cores.append(("#2ca25f" if "classe local" in n else "#99d8c9") if "fixa" in n.split(" | ")[1]
                         else ("#5e3c99" if "classe local" in n else "#bcbddc"))
    iou = [tabela["Todos"][n]["IoU"] for n in nomes]
    ax[0].barh(y, iou, color=cores, edgecolor="k", lw=0.4)
    for yy, v in zip(y, iou):
        ax[0].text(v + 0.005, yy, "%.3f" % v, va="center", fontsize=8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([n.replace(" | ", ", ") for n in nomes], fontsize=8)
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("IoU da água (todos os recortes)")
    ax[0].axvline(tabela["Todos"]["SOM por recorte"]["IoU"], color="#3fa9f5", ls="--", lw=1)
    ax[0].grid(axis="x", alpha=0.3)
    ax[0].set_title("Pré-treino (imagem, normalização, classe das unidades)")
    destaque = ["SOM por recorte", "3 imagens | fixa | classe fixa", "3 imagens | fixa | classe local",
                "3 imagens | percentil | classe local", "MNDWI"]
    for n in destaque:
        ax[1].plot([f for f, _, _ in faixas[1:]], [tabela[f][n]["IoU"] for f, _, _ in faixas[1:]], "o-", lw=2,
                   color=cores[nomes.index(n)], label=n.replace(" | ", ", "))
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Fração de água no recorte")
    ax[1].set_ylabel("IoU da água")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=9)
    ax[1].set_title("Por fração de água")
    fig.suptitle("Sen1Floods11: SOM pré-treinado em imagem balanceada e aplicado sem alteração aos outros %d recortes "
                 "(%d unidades, %d sorteios)" % (len(resultados), enchenteSOM.SOM_SIZE, args.sementes), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "pre_treino_som.png"), dpi=85)
    print("Resultados salvos em", args.saida)
