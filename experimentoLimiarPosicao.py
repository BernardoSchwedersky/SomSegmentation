# -*- coding: utf-8 -*-
"""
Avalia no Sen1Floods11 o SOM com limiar na posição da unidade (sem índice de
água na decisão), só na área alagada, como em experimentoAreaAlagada.py.

Método (enchenteSOM.montar_entradas_por_data, treinar_som_por_data,
classes_por_posicao): um único SOM 1D de 36 unidades é treinado com pixels das
duas datas, cada pixel descrito só pela sua data (bandas + MNDWI, janela 3x3,
normalização fixa). Cada pixel cai numa unidade antes e numa no evento. O limiar
é a posição k no grid: as unidades de um lado de k são água. Água nas duas datas
é água permanente, água só no evento é alagamento.

Pré-treino: uma imagem balanceada de cada evento (as mesmas dos experimentos
anteriores), juntas, com todos os eventos ou sem o evento avaliado. O limiar k e
o lado da água são escolhidos uma vez por SOM, nas imagens de pré-treino, pelo
maior IoU da água total contra o rótulo à mão (a única vez em que o rótulo
entra); depois o mesmo k vale para todos os recortes.

Também são guardadas as contagens para todos os k, para a curva IoU x k e o
melhor k por recorte (limite superior, que usa o rótulo do próprio recorte).

Uso:
    python experimentoLimiarPosicao.py --pasta-cache <cache> --sementes 3
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

import enchenteSOM as es
from avaliarSen1Floods11 import contar, metricas, somar
from experimentoAreaAlagada import ler_jrc
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from experimentoPreTreinoSOM import PRE_TREINO
from variacaoUnidadesSOM import ler_recorte

MODELOS = ["todos os eventos", "outros eventos"]
K = np.arange(es.SOM_SIZE + 1)


def resultados_por_data(SOM, antes, depois):
    ea, ed = es.montar_entradas_por_data(antes, depois)
    return es.segmentar(SOM, ea, 1), es.segmentar(SOM, ed, 1)

# Pré-treina o SOM e escolhe o limiar k e o lado da água nas imagens de pré-treino
def pre_treinar_posicao(recortes, seed, pasta_cache):
    dados = [ler_recorte({"recorte": r}, pasta_cache) for r in recortes]
    entradas = []
    for rotulo, antes, depois, invalidos in dados:
        ea, ed = es.montar_entradas_por_data(antes, depois)
        entradas += [(ea, invalidos), (ed, invalidos)]
    SOM = es.treinar_som_por_data(entradas, seed=seed)
    melhor = (-1, None, None)
    conts = {(k, lado): [] for k in K for lado in (True, False)}
    for (rotulo, antes, depois, invalidos), (ed, _) in zip(dados, entradas[1::2]):
        r = es.segmentar(SOM, ed, 1)
        v = (rotulo >= 0) & ~invalidos
        for (k, lado), l in conts.items():
            l.append(contar(es.agua_por_posicao(r, k, lado), rotulo == 1, v))
    for (k, lado), l in conts.items():
        iou = metricas(somar(l))["IoU"]
        if iou > melhor[0]:
            melhor = (iou, int(k), lado)
    return {"SOM": SOM, "k": melhor[1], "agua_acima": melhor[2], "iou_agua_pre_treino": melhor[0]}

def avaliar(args):
    r, modelos, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "limiar_posicao", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    jrc = ler_jrc(r["recorte"], pasta_cache)
    validos = (rotulo >= 0) & ~invalidos
    alagado_ref = (rotulo == 1) & ~jrc
    res = {"recorte": r["recorte"], "pais": r["pais"],
           "pct_alagado_ref": round(100 * float(alagado_ref[validos].mean()), 2), "modelos": {}}
    for nome, lista in modelos.items():
        res["modelos"][nome] = []
        for m in lista:
            ra, rd = resultados_por_data(m["SOM"], antes, depois)
            curva = [contar(es.classes_por_posicao(ra, rd, k, m["agua_acima"]) == 2, alagado_ref, validos) for k in K]
            mapa = es.classes_por_posicao(ra, rd, m["k"], m["agua_acima"])
            res["modelos"][nome].append({"k": m["k"], "alagamento": curva[m["k"]], "curva": curva,
                                         "permanente": contar(mapa == 1, jrc, validos)})
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s alagado %5.1f%% | " % (r["recorte"], res["pct_alagado_ref"]) + " ".join(
        "%s %.2f" % (n[:5], np.mean([metricas(s["alagamento"])["IoU"] for s in res["modelos"][n]])) for n in MODELOS),
        flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOM com limiar na posição da unidade, área alagada no Sen1Floods11")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.pasta_cache, "limiar_posicao"), exist_ok=True)

    por_evento = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    escolhidos = {p: escolher_pre_treino(l) for p, l in por_evento.items()}
    excluidos = set(r["recorte"] for r in escolhidos.values()) | set(PRE_TREINO)
    imagens = {p: e["recorte"] for p, e in escolhidos.items() if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX}

    todos = [pre_treinar_posicao(sorted(imagens.values()), s, args.pasta_cache) for s in range(args.sementes)]
    print("Pré-treino com todos os eventos: " + ", ".join(
        "k=%d (água %s), IoU água %.3f" % (m["k"], "acima" if m["agua_acima"] else "abaixo", m["iou_agua_pre_treino"])
        for m in todos), flush=True)
    tarefas, limiares = [], {"todos os eventos": [(m["k"], m["agua_acima"]) for m in todos]}
    for p, l in sorted(por_evento.items()):
        outros = sorted(r for q, r in imagens.items() if q != p)
        modelos = {"todos os eventos": todos,
                   "outros eventos": [pre_treinar_posicao(outros, s, args.pasta_cache) for s in range(args.sementes)]}
        limiares["outros eventos (%s)" % p] = [(m["k"], m["agua_acima"]) for m in modelos["outros eventos"]]
        tarefas += [(r, modelos, args.pasta_cache) for r in l if r["recorte"] not in excluidos]
        print("%-10s pré-treinos prontos: k = %s" % (p, limiares["outros eventos (%s)" % p]), flush=True)
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar, tarefas, chunksize=1)

    # Método anterior (limiar 0 no MNDWI médio da unidade) e MNDWI por pixel, nos mesmos recortes
    anteriores = {}
    for r in resultados:
        with open(os.path.join(args.pasta_cache, "area_alagada", r["recorte"] + ".json")) as f:
            anteriores[r["recorte"]] = json.load(f)
    def media_iou(listas_por_semente):
        return float(np.mean([metricas(somar(l)) ["IoU"] for l in listas_por_semente]))

    s = args.sementes
    tabela = {}
    for n in MODELOS:
        ms = [metricas(somar([r["modelos"][n][i]["alagamento"] for r in resultados])) for i in range(s)]
        perm = [metricas(somar([r["modelos"][n][i]["permanente"] for r in resultados])) for i in range(s)]
        curva = [[metricas(somar([r["modelos"][n][i]["curva"][k] for r in resultados]))["IoU"] for k in K]
                 for i in range(s)]
        oraculo = [metricas(somar([max(r["modelos"][n][i]["curva"], key=lambda c: metricas(c)["IoU"])
                                   for r in resultados]))["IoU"] for i in range(s)]
        tabela["SOM posição, pré-treino " + n] = {
            **{m: float(np.mean([x[m] for x in ms])) for m in ms[0]},
            "IoU_permanente": float(np.mean([x["IoU"] for x in perm])),
            "IoU_melhor_k_global": float(np.mean([max(c) for c in curva])),
            "melhor_k_global": [int(np.argmax(c)) for c in curva],
            "IoU_melhor_k_por_recorte": float(np.mean(oraculo)),
            "curva_IoU": [float(x) for x in np.mean(curva, axis=0)]}
    for n_ant, rotulo in [("SOM pré-treinado (todos os eventos)", "SOM MNDWI da unidade, pré-treino todos os eventos"),
                          ("SOM pré-treinado (outros eventos)", "SOM MNDWI da unidade, pré-treino outros eventos"),
                          ("MNDWI", "MNDWI por pixel")]:
        k = len(anteriores[resultados[0]["recorte"]]["alagamento"][n_ant])
        ms = [metricas(somar([anteriores[r["recorte"]]["alagamento"][n_ant][i] for r in resultados])) for i in range(k)]
        perm = [metricas(somar([anteriores[r["recorte"]]["permanente"][n_ant][i] for r in resultados]))
                for i in range(k)]
        tabela[rotulo] = {**{m: float(np.mean([x[m] for x in ms])) for m in ms[0]},
                          "IoU_permanente": float(np.mean([x["IoU"] for x in perm]))}

    eventos = sorted(por_evento)
    por_ev = {p: {n: float(np.mean([metricas(somar([r["modelos"][n][i]["alagamento"] for r in resultados
                                                     if r["pais"] == p]))["IoU"] for i in range(s)]))
                  for n in MODELOS} for p in eventos}
    for p in eventos:
        por_ev[p]["MNDWI"] = metricas(somar([anteriores[r["recorte"]]["alagamento"]["MNDWI"][0]
                                             for r in resultados if r["pais"] == p]))["IoU"]
        por_ev[p]["SOM MNDWI da unidade"] = float(np.mean([metricas(somar(
            [anteriores[r["recorte"]]["alagamento"]["SOM pré-treinado (todos os eventos)"][i]
             for r in resultados if r["pais"] == p]))["IoU"] for i in range(s)]))

    print("\n%d recortes; IoU do alagamento (classe 2 x rótulo que não é água permanente no JRC)" % len(resultados))
    print("%-52s %6s %6s %6s %6s %8s" % ("Método", "IoU", "F1", "prec.", "sens.", "IoU perm."))
    for n, t in tabela.items():
        print("%-52s %6.3f %6.3f %6.3f %6.3f %8.3f" % (n, t["IoU"], t["F1"], t["precisao"], t["sensibilidade"],
                                                     t["IoU_permanente"]))
    for n in MODELOS:
        t = tabela["SOM posição, pré-treino " + n]
        print("%s: melhor k global %s -> IoU %.3f; melhor k em cada recorte -> IoU %.3f" % (
            n, t["melhor_k_global"], t["IoU_melhor_k_global"], t["IoU_melhor_k_por_recorte"]))
    print("\n%-10s %10s %10s %12s %8s" % ("Evento", "posição", "pos.outros", "MNDWI unid.", "MNDWI"))
    for p in eventos:
        e = por_ev[p]
        print("%-10s %10.3f %10.3f %12.3f %8.3f" % (p, e["todos os eventos"], e["outros eventos"],
                                                   e["SOM MNDWI da unidade"], e["MNDWI"]))

    with open(os.path.join(args.saida, "limiar_posicao.json"), "w") as f:
        json.dump({"n_recortes": len(resultados), "limiares_pre_treino": limiares, "tabela": tabela,
                   "por_evento": por_ev}, f, indent=1, ensure_ascii=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, cor in zip(MODELOS, ["#1f77b4", "#ff7f0e"]):
        t = tabela["SOM posição, pré-treino " + n]
        ax[0].plot(K, t["curva_IoU"], color=cor, label="SOM posição, pré-treino " + n)
    for n, cor, est in [("SOM MNDWI da unidade, pré-treino todos os eventos", "#555555", "--"),
                        ("MNDWI por pixel", "#d62728", ":")]:
        ax[0].axhline(tabela[n]["IoU"], color=cor, ls=est, label=n)
    for k, lado in limiares["todos os eventos"]:
        ax[0].axvline(k, color="#1f77b4", alpha=0.3, lw=3)
    ax[0].set_xlabel("limiar k (posição da unidade no SOM)")
    ax[0].set_ylabel("IoU do alagamento (todos os recortes)")
    ax[0].set_title("IoU x limiar de posição (faixa: k escolhido no pré-treino)")
    ax[0].legend(fontsize=8)
    x = np.arange(len(eventos))
    for i, (n, cor) in enumerate([("todos os eventos", "#1f77b4"), ("outros eventos", "#ff7f0e"),
                                  ("SOM MNDWI da unidade", "#555555"), ("MNDWI", "#d62728")]):
        ax[1].bar(x + (i - 1.5) * 0.2, [por_ev[p][n] for p in eventos], 0.2, color=cor,
                  label={"todos os eventos": "posição (todos)", "outros eventos": "posição (outros)"}.get(n, n))
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(eventos, rotation=45, ha="right")
    ax[1].set_ylabel("IoU do alagamento")
    ax[1].set_title("Por evento")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "limiar_posicao.png"), dpi=130)
    print("Figura salva em", os.path.join(args.saida, "limiar_posicao.png"))
