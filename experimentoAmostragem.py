# -*- coding: utf-8 -*-
"""
Testa a variação do SOM bi-temporal que treina com uma porcentagem dos pixels
sorteada aleatoriamente (enchenteSOM.treinar_e_segmentar_amostrado): tempo de
treinamento e resultado contra os rótulos à mão do Sen1Floods11.

Usa os recortes e as cenas já escolhidos por avaliarSen1Floods11.py (mesma pasta
de cache). Sorteia recortes estratificados pela fração de água do rótulo e, para
cada porcentagem e cada semente do sorteio, treina o SOM, segmenta o recorte e
mede: tempo de treino, IoU da água contra o rótulo e concordância com o SOM
treinado com 100% dos pixels (mesma semente).

Saída: resultados/amostragem/amostragem.json e amostragem.png.

Os resultados em resultados/amostragem/ usam 12 unidades (o padrão na época).

Uso:
    python experimentoAmostragem.py --porcentagens 1 2 5 10 20 50 100 --sementes 5 --recortes 40
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


def rodar(args):
    r, porcentagens, sementes, pasta_cache = args
    try:
        rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    except Exception as erro:
        print("%-22s erro: %s" % (r["recorte"], erro), flush=True)
        return None
    validos = (rotulo >= 0) & ~invalidos
    referencia = rotulo == 1
    entrada, _ = enchenteSOM.montar_entrada(antes, depois)
    res = {"recorte": r["recorte"], "pct_agua_rotulo": r["pct_agua_rotulo"], "rodadas": []}
    agua_100 = {}
    for p in sorted(porcentagens, reverse=True):  # 100% primeiro, para a concordância
        for s in range(sementes):
            _, result, tempos = enchenteSOM.treinar_e_segmentar_amostrado(entrada, invalidos, p / 100, som_size=12, seed=s)
            classe_unidade, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, 12)
            agua = np.isin(classe_unidade[result], [1, 2])
            if p == max(porcentagens):
                agua_100[s] = agua
            res["rodadas"].append({"porcentagem": p, "semente": s, **tempos, "contagens": contar(agua, referencia, validos),
                                   "concordancia_100": float((agua[validos] == agua_100[s][validos]).mean())})
    print("%-22s água %5.1f%% | " % (r["recorte"], r["pct_agua_rotulo"]) + " ".join(
        "%g%%:%.2f" % (p, np.mean([metricas(x["contagens"])["IoU"] for x in res["rodadas"] if x["porcentagem"] == p]))
        for p in porcentagens), flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o treino do SOM com uma porcentagem sorteada dos pixels")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "amostragem"))
    parser.add_argument("--porcentagens", nargs="+", type=float, default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--sementes", type=int, default=5, help="Sorteios diferentes por porcentagem")
    parser.add_argument("--recortes", type=int, default=40, help="Recortes sorteados (estratificados pela água)")
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.saida, exist_ok=True)
    os.makedirs(os.path.join(args.pasta_cache, "bandas"), exist_ok=True)

    todos = []
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            todos.append(r)
    # Sorteio estratificado pela fração de água do rótulo (mesmo número por faixa)
    faixas = [(0, 1), (1, 5), (5, 15), (15, 40), (40, 101)]
    rng = np.random.RandomState(0)
    recortes = []
    for a, b in faixas:
        sel = [r for r in todos if a <= r["pct_agua_rotulo"] < b]
        recortes += [sel[i] for i in rng.choice(len(sel), min(len(sel), args.recortes // len(faixas)), replace=False)]
    print("%d recortes, porcentagens %s, %d sementes" % (len(recortes), args.porcentagens, args.sementes), flush=True)
    with Pool(args.processos) as pool:
        resultados = [r for r in pool.map(rodar, [(r, args.porcentagens, args.sementes, args.pasta_cache)
                                                    for r in recortes], chunksize=1) if r]

    # Resumo por porcentagem: IoU agregada (por semente), tempo médio e concordância com 100%
    resumo = {}
    for p in args.porcentagens:
        rod = [(r, x) for r in resultados for x in r["rodadas"] if x["porcentagem"] == p]
        iou_semente = [metricas(somar([x["contagens"] for r, x in rod if x["semente"] == s]))["IoU"]
                       for s in range(args.sementes)]
        # Variação entre sementes dentro de cada recorte (desvio padrão da IoU)
        dp_recorte = [np.std([metricas(x["contagens"])["IoU"] for x in r["rodadas"] if x["porcentagem"] == p])
                      for r in resultados if r["pct_agua_rotulo"] >= 1]
        resumo[p] = {"amostras_media": float(np.mean([x["amostras"] for _, x in rod])),
                     "treino_s": float(np.mean([x["treino_s"] for _, x in rod])),
                     "segmentacao_s": float(np.mean([x["segmentacao_s"] for _, x in rod])),
                     "iou_media": float(np.mean(iou_semente)), "iou_min": float(np.min(iou_semente)),
                     "iou_max": float(np.max(iou_semente)),
                     "dp_iou_entre_sementes_por_recorte": float(np.mean(dp_recorte)),
                     "concordancia_100": float(np.mean([x["concordancia_100"] for _, x in rod])),
                     **{"iou_agua_%d-%d%%" % (a, min(b, 100)): float(np.mean([metricas(somar(
                         [x["contagens"] for r, x in rod if x["semente"] == s and a <= r["pct_agua_rotulo"] < b]))["IoU"]
                         for s in range(args.sementes)])) for a, b in faixas}}

    print("\n%d recortes avaliados" % len(resultados))
    print("%6s %9s %9s %9s %8s %9s %10s %12s" % ("%", "amostras", "treino s", "segm. s", "IoU", "IoU min", "dp/recorte",
                                                 "concord.100%"))
    for p, v in resumo.items():
        print("%6g %9.0f %9.2f %9.2f %8.3f %9.3f %10.3f %12.3f" % (p, v["amostras_media"], v["treino_s"], v["segmentacao_s"],
              v["iou_media"], v["iou_min"], v["dp_iou_entre_sementes_por_recorte"], v["concordancia_100"]))
    with open(os.path.join(args.saida, "amostragem.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "sementes": args.sementes, "resumo": resumo,
                   "por_recorte": [{"recorte": r["recorte"], "pct_agua_rotulo": r["pct_agua_rotulo"]} for r in resultados]},
                  f, ensure_ascii=False, indent=1)

    # Figura: tempo de treino, IoU (com faixa entre sementes) e concordância com 100%
    ps = list(resumo)
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.5))
    ax[0].plot(ps, [resumo[p]["treino_s"] for p in ps], "o-", lw=2.5, color="#3fa9f5", label="Treino (sorteio + SOM)")
    ax[0].plot(ps, [resumo[p]["segmentacao_s"] for p in ps], "s--", lw=2, color="#7f7f7f", label="Segmentação (todos os pixels)")
    ax[0].set_ylabel("Tempo médio por recorte (s)")
    ax[0].set_title("Tempo (recortes de 512x512 pixels)")
    ax[1].plot(ps, [resumo[p]["iou_media"] for p in ps], "o-", lw=2.5, color="#3fa9f5", label="Todos os recortes")
    ax[1].fill_between(ps, [resumo[p]["iou_min"] for p in ps], [resumo[p]["iou_max"] for p in ps], color="#3fa9f5", alpha=0.2)
    for (a, b), cor in zip(faixas[1:], ["#fdae61", "#abdda4", "#2b83ba", "#5e3c99"]):
        chave = "iou_agua_%d-%d%%" % (a, min(b, 100))
        ax[1].plot(ps, [resumo[p][chave] for p in ps], ".--", color=cor, lw=1.3, label="%d-%d%% de água" % (a, min(b, 100)))
    ax[1].set_ylabel("IoU da água (rótulo à mão)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Resultado (faixa = entre %d sorteios)" % args.sementes)
    ax[2].plot(ps, [resumo[p]["concordancia_100"] for p in ps], "o-", lw=2.5, color="#3fa9f5", label="Concordância com 100%")
    ax[2].plot(ps, [resumo[p]["dp_iou_entre_sementes_por_recorte"] for p in ps], "s--", lw=2, color="#d95f0e",
               label="Desvio da IoU entre sorteios (por recorte)")
    ax[2].set_title("Estabilidade")
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(ps)
        a.set_xticklabels(["%g%%" % p for p in ps])
        a.set_xlabel("Porcentagem de pixels usada no treino")
        a.grid(alpha=0.3)
        a.legend(fontsize=9)
    fig.suptitle("SOM treinado com uma porcentagem sorteada dos pixels - %d recortes do Sen1Floods11" % len(resultados),
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "amostragem.png"), dpi=85)
    print("Resultados salvos em", args.saida)
