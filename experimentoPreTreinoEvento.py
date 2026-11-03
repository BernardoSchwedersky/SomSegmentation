# -*- coding: utf-8 -*-
"""
Testa o pré-treino do SOM com uma imagem do próprio evento: para cada evento do
Sen1Floods11, o SOM é pré-treinado no recorte com a fração de água mais perto
de 50% (entre os com pelo menos 80% de pixels válidos) e aplicado, sem
alteração, aos outros recortes do mesmo evento.

Compara com o pré-treino em imagens de eventos distintos
(experimentoPreTreinoSOM.py, "3 imagens") e com o SOM treinado em cada recorte
(experimentoFeaturesCNN.py), nos mesmos recortes: os de pré-treino de todos os
eventos ficam fora da avaliação. Os dois scripts precisam ter rodado antes, com
a mesma pasta de cache.

Uso:
    python experimentoPreTreinoEvento.py --sementes 3
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
from experimentoPreTreinoSOM import NORMALIZACOES, PRE_TREINO, pre_treinar
from variacaoUnidadesSOM import ler_recorte


def escolher_pre_treino(recortes):
    candidatos = [r for r in recortes if r["pct_validos"] >= 80] or recortes
    return min(candidatos, key=lambda r: abs(r["pct_agua_rotulo"] - 50))

def avaliar(args):
    r, modelos, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "pre_treino_evento", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    validos = (rotulo >= 0) & ~invalidos
    referencia = rotulo == 1
    res = {"recorte": r["recorte"], "pais": r["pais"], "pct_agua_rotulo": r["pct_agua_rotulo"], "variacoes": {}}
    for norm in NORMALIZACOES:
        entrada, _ = enchenteSOM.montar_entrada(antes, depois, normalizacao=norm)
        for SOM, classe_fixa in modelos[norm]:
            result = enchenteSOM.segmentar(SOM, entrada, 1)
            classe_local, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, enchenteSOM.SOM_SIZE)
            for tipo, classe in (("classe fixa", classe_fixa), ("classe local", classe_local)):
                res["variacoes"].setdefault("Evento | %s | %s" % (norm, tipo), []).append(
                    contar(np.isin(classe[result], [1, 2]), referencia, validos))
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s água %5.1f%%" % (r["recorte"], r["pct_agua_rotulo"]), flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o SOM pré-treinado em uma imagem do próprio evento")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.pasta_cache, "pre_treino_evento"), exist_ok=True)

    por_evento = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    escolhidos = {p: escolher_pre_treino(l) for p, l in por_evento.items()}
    excluidos = set(r["recorte"] for r in escolhidos.values()) | set(PRE_TREINO)

    tarefas = []
    for p, l in sorted(por_evento.items()):
        e = escolhidos[p]
        modelos = {norm: [pre_treinar([e["recorte"]], norm, s, args.pasta_cache) for s in range(args.sementes)]
                   for norm in NORMALIZACOES}
        print("%-10s pré-treino %-18s (água %4.1f%%): unidades de água por semente %s" % (
            p, e["recorte"], e["pct_agua_rotulo"], [int(np.isin(c, [1, 2]).sum()) for _, c in modelos["fixa"]]), flush=True)
        tarefas += [(r, modelos, args.pasta_cache) for r in l if r["recorte"] not in excluidos]
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar, tarefas, chunksize=1)

    # Referências nos mesmos recortes
    def ler(pasta, rec):
        with open(os.path.join(args.pasta_cache, pasta, rec + ".json")) as f:
            return json.load(f)
    entre = {r["recorte"]: ler("pre_treino", r["recorte"]) for r in resultados}
    proprio = {r["recorte"]: ler("features_cnn", r["recorte"]) for r in resultados}
    variacoes_evento = list(resultados[0]["variacoes"])
    variacoes_entre = ["3 imagens | %s | %s" % (n, t) for n in NORMALIZACOES for t in ("classe fixa", "classe local")]
    nomes = ["SOM por recorte"] + variacoes_entre + variacoes_evento + ["NDWI", "MNDWI"]

    def contagens(r, n, s):
        if n == "SOM por recorte":
            return proprio[r["recorte"]]["SOM"]["SOM"][s]
        if n in ("NDWI", "MNDWI"):
            return proprio[r["recorte"]][n]
        if n.startswith("3 imagens"):
            return entre[r["recorte"]]["variacoes"][n][s]
        return r["variacoes"][n][s]

    def tabela_de(sel):
        t = {"n": len(sel)}
        for n in nomes:
            ms = [metricas(somar([contagens(r, n, s) for r in sel])) for s in range(args.sementes)]
            t[n] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        return t

    faixas = [("Todos", 0, 101), ("0-1%", 0, 1), ("1-5%", 1, 5), ("5-15%", 5, 15), ("15-40%", 15, 40), ("40-100%", 40, 101)]
    tabela = {f: tabela_de([r for r in resultados if a <= r["pct_agua_rotulo"] < b]) for f, a, b in faixas}
    eventos = sorted(por_evento)
    tabela_eventos = {p: tabela_de([r for r in resultados if r["pais"] == p]) for p in eventos}
    com_agua = [r for r in resultados if r["pct_agua_rotulo"] >= 1]
    falhas = {n: sum(1 for r in com_agua if np.mean([metricas(contagens(r, n, s))["IoU"] for s in range(args.sementes)])
                     < 0.05) for n in nomes}

    print("\n%d recortes avaliados (sem os %d de pré-treino)" % (len(resultados), len(excluidos)))
    print("%-36s %6s %6s %6s %6s %6s %6s %6s %6s %7s" % ("Variação", "Todos", "0-1%", "1-5%", "5-15%", "15-40%", ">40%",
                                                       "prec.", "sens.", "falhas"))
    for n in nomes:
        print("%-36s " % n + " ".join("%6.3f" % tabela[f][n]["IoU"] for f, _, _ in faixas)
              + " %6.3f %6.3f %7d" % (tabela["Todos"][n]["precisao"], tabela["Todos"][n]["sensibilidade"], falhas[n]))
    principais = ["SOM por recorte", "3 imagens | fixa | classe local", "Evento | fixa | classe local",
                  "Evento | fixa | classe fixa", "Evento | percentil | classe local", "MNDWI"]
    print("\nIoU por evento:")
    print("%-10s %4s %-20s " % ("Evento", "n", "pré-treino (água)") + " ".join("%13s" % n.split(" | ")[0][:6]
          + ("/" + n.split(" | ")[1][:4] + "/" + n.split(" | ")[2][7:10] if " | " in n else "") for n in principais))
    for p in eventos:
        print("%-10s %4d %-20s " % (p, tabela_eventos[p]["n"], "%s (%.0f%%)" % (escolhidos[p]["recorte"].split("_")[1],
              escolhidos[p]["pct_agua_rotulo"])) + " ".join("%13.3f" % tabela_eventos[p][n]["IoU"] for n in principais))

    with open(os.path.join(args.saida, "pre_treino_evento.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "sementes": args.sementes, "unidades": enchenteSOM.SOM_SIZE,
                   "pre_treino_por_evento": {p: {"recorte": e["recorte"], "pct_agua_rotulo": e["pct_agua_rotulo"]}
                                             for p, e in escolhidos.items()},
                   "metricas": tabela, "metricas_por_evento": tabela_eventos, "falhas_recortes_com_agua": falhas,
                   "recortes_com_agua": len(com_agua)}, f, ensure_ascii=False, indent=1)

    # Figura: IoU por evento das variações principais e IoU por fração de água
    cores = {"SOM por recorte": "#3fa9f5", "3 imagens | fixa | classe local": "#2ca25f",
             "Evento | fixa | classe local": "#d95f0e", "Evento | fixa | classe fixa": "#fdae6b",
             "Evento | percentil | classe local": "#8c6bb1", "MNDWI": "#ff9f1c"}
    rotulos = {"SOM por recorte": "SOM treinado em cada recorte", "3 imagens | fixa | classe local":
               "Pré-treino em 3 imagens de outros eventos (fixa, classe local)",
               "Evento | fixa | classe local": "Pré-treino no próprio evento (fixa, classe local)",
               "Evento | fixa | classe fixa": "Pré-treino no próprio evento (fixa, classe fixa)",
               "Evento | percentil | classe local": "Pré-treino no próprio evento (percentil, classe local)",
               "MNDWI": "MNDWI"}
    fig, ax = plt.subplots(1, 2, figsize=(24, 6.5), gridspec_kw=dict(width_ratios=[2.3, 1]))
    grupos = ["Todos"] + eventos
    w = 0.8 / len(principais)
    for k, n in enumerate(principais):
        v = [tabela["Todos"][n]["IoU"]] + [tabela_eventos[p][n]["IoU"] for p in eventos]
        ax[0].bar(np.arange(len(grupos)) + (k - (len(principais) - 1) / 2) * w, v, w, color=cores[n], label=rotulos[n],
                  edgecolor="k", lw=0.4)
    ax[0].set_xticks(np.arange(len(grupos)))
    ax[0].set_xticklabels(["Todos\n(%d)" % len(resultados)] + ["%s\n(%d; %.0f%%)" % (p, tabela_eventos[p]["n"],
                           escolhidos[p]["pct_agua_rotulo"]) for p in eventos], fontsize=9)
    ax[0].set_ylabel("IoU da água")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    ax[0].set_title("Por evento (recortes avaliados; água no recorte de pré-treino do evento)")
    for n in principais:
        ax[1].plot([f for f, _, _ in faixas[1:]], [tabela[f][n]["IoU"] for f, _, _ in faixas[1:]], "o-", lw=2,
                   color=cores[n], label=rotulos[n])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("Fração de água no recorte")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Por fração de água")
    fig.suptitle("Sen1Floods11: SOM pré-treinado em uma imagem do próprio evento x em imagens de outros eventos "
                 "(%d unidades, %d sorteios)" % (enchenteSOM.SOM_SIZE, args.sementes), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "pre_treino_evento.png"), dpi=85)
    print("Resultados salvos em", args.saida)
