# -*- coding: utf-8 -*-
"""
Testa o pré-treino do SOM com uma imagem balanceada de cada evento do
Sen1Floods11, todas juntas no mesmo SOM, aplicado sem alteração aos demais
recortes.

As imagens de pré-treino são as escolhidas por experimentoPreTreinoEvento.py (o
recorte de cada evento com a fração de água mais perto de 50%), só as que têm
entre 20% e 80% de água. Duas versões, com normalização fixa:
- "Todos os eventos": um único SOM pré-treinado com as imagens de todos os
  eventos;
- "Outros eventos": para cada evento avaliado, um SOM pré-treinado só com as
  imagens dos demais eventos (mede a generalização para um evento novo).
As unidades são classificadas com a classe do pré-treino ("classe fixa") ou
recalculada em cada recorte pelo MNDWI ("classe local").

A avaliação usa os mesmos recortes de experimentoPreTreinoEvento.py (sem os de
pré-treino) e compara com os resultados guardados por ele,
experimentoPreTreinoSOM.py e experimentoFeaturesCNN.py (mesma pasta de cache).

Uso:
    python experimentoPreTreinoMultiEvento.py --sementes 3
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import enchenteSOM
from avaliarSen1Floods11 import contar, metricas, somar
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoSOM import PRE_TREINO, pre_treinar
from variacaoUnidadesSOM import ler_recorte

AGUA_MIN, AGUA_MAX = 20, 80  # Fração de água (%) aceita numa imagem de pré-treino


def avaliar(args):
    r, modelos, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "pre_treino_multi", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    validos = (rotulo >= 0) & ~invalidos
    referencia = rotulo == 1
    entrada, _ = enchenteSOM.montar_entrada(antes, depois, normalizacao="fixa")
    res = {"recorte": r["recorte"], "pais": r["pais"], "pct_agua_rotulo": r["pct_agua_rotulo"], "variacoes": {}}
    for nome, lista in modelos.items():
        for SOM, classe_fixa in lista:
            result = enchenteSOM.segmentar(SOM, entrada, 1)
            classe_local, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, enchenteSOM.SOM_SIZE)
            for tipo, classe in (("classe fixa", classe_fixa), ("classe local", classe_local)):
                res["variacoes"].setdefault("%s | fixa | %s" % (nome, tipo), []).append(
                    contar(np.isin(classe[result], [1, 2]), referencia, validos))
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s água %5.1f%%" % (r["recorte"], r["pct_agua_rotulo"]), flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o SOM pré-treinado com uma imagem de cada evento juntas")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.pasta_cache, "pre_treino_multi"), exist_ok=True)

    por_evento = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    escolhidos = {p: escolher_pre_treino(l) for p, l in por_evento.items()}
    excluidos = set(r["recorte"] for r in escolhidos.values()) | set(PRE_TREINO)  # Mesmos recortes do teste anterior
    imagens = {p: e["recorte"] for p, e in escolhidos.items() if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX}
    print("Imagens de pré-treino (%d eventos): %s" % (len(imagens), ", ".join("%s (%.0f%%)" % (r, escolhidos[p]["pct_agua_rotulo"])
                                                                           for p, r in sorted(imagens.items()))), flush=True)

    todos = [pre_treinar(sorted(imagens.values()), "fixa", s, args.pasta_cache) for s in range(args.sementes)]
    print("Todos os eventos: unidades de água por semente %s" % [int(np.isin(c, [1, 2]).sum()) for _, c in todos], flush=True)
    tarefas = []
    for p, l in sorted(por_evento.items()):
        outros = sorted(r for q, r in imagens.items() if q != p)
        modelos = {"Todos os eventos": todos,
                   "Outros eventos": [pre_treinar(outros, "fixa", s, args.pasta_cache) for s in range(args.sementes)]}
        print("%-10s outros eventos (%d imagens): unidades de água por semente %s" % (
            p, len(outros), [int(np.isin(c, [1, 2]).sum()) for _, c in modelos["Outros eventos"]]), flush=True)
        tarefas += [(r, modelos, args.pasta_cache) for r in l if r["recorte"] not in excluidos]
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar, tarefas, chunksize=1)

    # Resultados anteriores nos mesmos recortes
    def ler(pasta, rec):
        with open(os.path.join(args.pasta_cache, pasta, rec + ".json")) as f:
            return json.load(f)
    anteriores = {r["recorte"]: {"evento": ler("pre_treino_evento", r["recorte"]), "3": ler("pre_treino", r["recorte"]),
                                 "proprio": ler("features_cnn", r["recorte"])} for r in resultados}
    novas = list(resultados[0]["variacoes"])
    nomes = ["SOM por recorte", "3 imagens | fixa | classe local", "3 imagens | fixa | classe fixa",
             "Evento | fixa | classe local", "Evento | fixa | classe fixa"] + novas + ["NDWI", "MNDWI"]

    def contagens(r, n, s):
        a = anteriores[r["recorte"]]
        if n == "SOM por recorte":
            return a["proprio"]["SOM"]["SOM"][s]
        if n in ("NDWI", "MNDWI"):
            return a["proprio"][n]
        if n.startswith("3 imagens"):
            return a["3"]["variacoes"][n][s]
        if n.startswith("Evento"):
            return a["evento"]["variacoes"][n][s]
        return r["variacoes"][n][s]

    def tabela_de(sel):
        t = {"n": len(sel)}
        for n in nomes:
            ms = [metricas(somar([contagens(r, n, s) for r in sel])) for s in range(args.sementes)]
            t[n] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            t[n]["IoU_min_semente"] = float(min(m["IoU"] for m in ms))
        return t

    faixas = [("Todos", 0, 101), ("0-1%", 0, 1), ("1-5%", 1, 5), ("5-15%", 5, 15), ("15-40%", 15, 40), ("40-100%", 40, 101)]
    tabela = {f: tabela_de([r for r in resultados if a <= r["pct_agua_rotulo"] < b]) for f, a, b in faixas}
    eventos = sorted(por_evento)
    tabela_eventos = {p: tabela_de([r for r in resultados if r["pais"] == p]) for p in eventos}
    com_agua = [r for r in resultados if r["pct_agua_rotulo"] >= 1]
    falhas = {n: sum(1 for r in com_agua if np.mean([metricas(contagens(r, n, s))["IoU"] for s in range(args.sementes)])
                     < 0.05) for n in nomes}

    print("\n%d recortes avaliados" % len(resultados))
    print("%-40s %6s %6s %6s %6s %6s %6s %6s %6s %7s" % ("Variação", "Todos", "0-1%", "1-5%", "5-15%", "15-40%", ">40%",
                                                       "prec.", "sens.", "falhas"))
    for n in nomes:
        print("%-40s " % n + " ".join("%6.3f" % tabela[f][n]["IoU"] for f, _, _ in faixas)
              + " %6.3f %6.3f %7d" % (tabela["Todos"][n]["precisao"], tabela["Todos"][n]["sensibilidade"], falhas[n]))
    principais = ["SOM por recorte", "3 imagens | fixa | classe local", "Evento | fixa | classe local",
                  "Todos os eventos | fixa | classe local", "Outros eventos | fixa | classe local", "MNDWI"]
    print("\nIoU por evento:")
    print("%-10s %4s " % ("Evento", "n") + " ".join("%12s" % n.split(" | ")[0][:12] for n in principais))
    for p in eventos:
        print("%-10s %4d " % (p, tabela_eventos[p]["n"]) + " ".join("%12.3f" % tabela_eventos[p][n]["IoU"] for n in principais))

    with open(os.path.join(args.saida, "pre_treino_multi_evento.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "sementes": args.sementes, "unidades": enchenteSOM.SOM_SIZE,
                   "imagens_pre_treino": imagens, "metricas": tabela, "metricas_por_evento": tabela_eventos,
                   "falhas_recortes_com_agua": falhas, "recortes_com_agua": len(com_agua)}, f, ensure_ascii=False, indent=1)

    cores = {"SOM por recorte": "#3fa9f5", "3 imagens | fixa | classe local": "#2ca25f",
             "Evento | fixa | classe local": "#d95f0e", "Todos os eventos | fixa | classe local": "#6a51a3",
             "Outros eventos | fixa | classe local": "#bcbddc", "MNDWI": "#ff9f1c"}
    rotulos = {"SOM por recorte": "SOM treinado em cada recorte",
               "3 imagens | fixa | classe local": "Pré-treino em 3 imagens (Índia, Mekong, EUA)",
               "Evento | fixa | classe local": "Pré-treino em 1 imagem do próprio evento",
               "Todos os eventos | fixa | classe local": "Pré-treino em 1 imagem de cada evento (todos)",
               "Outros eventos | fixa | classe local": "Pré-treino em 1 imagem de cada evento (sem o avaliado)",
               "MNDWI": "MNDWI"}
    fig, ax = plt.subplots(1, 2, figsize=(24, 6.5), gridspec_kw=dict(width_ratios=[2.3, 1]))
    grupos = ["Todos"] + eventos
    w = 0.8 / len(principais)
    for k, n in enumerate(principais):
        v = [tabela["Todos"][n]["IoU"]] + [tabela_eventos[p][n]["IoU"] for p in eventos]
        ax[0].bar(np.arange(len(grupos)) + (k - (len(principais) - 1) / 2) * w, v, w, color=cores[n], label=rotulos[n],
                  edgecolor="k", lw=0.4)
    ax[0].set_xticks(np.arange(len(grupos)))
    ax[0].set_xticklabels(["Todos\n(%d)" % len(resultados)] + ["%s\n(%d)" % (p, tabela_eventos[p]["n"]) for p in eventos])
    ax[0].set_ylabel("IoU da água (classe local, normalização fixa)")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    ax[0].set_title("Por evento (número de recortes)")
    for n in principais:
        ax[1].plot([f for f, _, _ in faixas[1:]], [tabela[f][n]["IoU"] for f, _, _ in faixas[1:]], "o-", lw=2,
                   color=cores[n], label=rotulos[n])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Fração de água no recorte")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Por fração de água")
    fig.suptitle("Sen1Floods11: SOM pré-treinado com uma imagem balanceada de cada evento, juntas (%d imagens; "
                 "%d unidades, %d sorteios)" % (len(imagens), enchenteSOM.SOM_SIZE, args.sementes), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "pre_treino_multi_evento.png"), dpi=85)
    print("Resultados salvos em", args.saida)
