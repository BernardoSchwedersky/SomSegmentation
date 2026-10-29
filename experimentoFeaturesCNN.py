# -*- coding: utf-8 -*-
"""
Testa a inclusão de features de filtros convolucionais pré-treinados (VGG16 do
ImageNet, featuresCNN.py) na entrada do SOM bi-temporal, contra os rótulos à
mão do Sen1Floods11.

Variações (todas com o padrão de enchenteSOM: 36 unidades, treino com 2%):
- "SOM": bandas + MNDWI de cada data (entrada atual).
- "SOM + VGG k": a mesma entrada mais k componentes principais das features VGG
  de cada data.
As unidades continuam sendo classificadas pelo MNDWI médio, então as features só
mudam o agrupamento dos pixels.

Usa os recortes e as bandas já guardados por avaliarSen1Floods11.py e
variacaoUnidadesSOM.py (mesma pasta de cache).

Uso:
    python experimentoFeaturesCNN.py --componentes 4 8 --sementes 3
"""

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import enchenteSOM
import featuresCNN
from avaliarSen1Floods11 import contar, metricas, somar
from variacaoUnidadesSOM import ler_recorte


def rodar(args):
    r, componentes, sementes, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "features_cnn", r["recorte"] + ".json")
    variacoes = ["SOM"] + ["SOM + VGG %d" % k for k in componentes]
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            feito = json.load(f)
        if all(v in feito["SOM"] for v in variacoes):
            return feito
    try:
        rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    except Exception as erro:
        print("%-22s erro: %s" % (r["recorte"], erro), flush=True)
        return None
    validos = (rotulo >= 0) & ~invalidos
    referencia = rotulo == 1
    t0 = time.perf_counter()
    f_antes, f_depois = featuresCNN.features_pca(antes, depois, invalidos, max(componentes))
    t_features = time.perf_counter() - t0
    res = {"recorte": r["recorte"], "pais": r["pais"], "pct_agua_rotulo": r["pct_agua_rotulo"], "SOM": {}, "tempo": {},
           "tempo_features_s": t_features,
           "MNDWI": contar(np.isin(enchenteSOM.mapa_indice(antes, depois, 0, "MNDWI"), [1, 2]), referencia, validos),
           "NDWI": contar(np.isin(enchenteSOM.mapa_indice(antes, depois, 0, "NDWI"), [1, 2]), referencia, validos)}
    for v in variacoes:
        k = int(v.split()[-1]) if "VGG" in v else 0
        entrada, _ = enchenteSOM.montar_entrada(antes, depois, extras=(f_antes[:, :, :k], f_depois[:, :, :k]) if k else None)
        res["SOM"][v], res["tempo"][v] = [], []
        for s in range(sementes):
            _, result, tempos = enchenteSOM.treinar_e_segmentar_amostrado(entrada, invalidos, seed=s)
            classe_unidade, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, enchenteSOM.SOM_SIZE)
            res["SOM"][v].append(contar(np.isin(classe_unidade[result], [1, 2]), referencia, validos))
            res["tempo"][v].append(tempos["treino_s"] + tempos["segmentacao_s"])
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s água %5.1f%% | " % (r["recorte"], r["pct_agua_rotulo"]) + " ".join(
        "%s %.2f" % (v.replace("SOM + ", "+"), metricas(somar(res["SOM"][v]))["IoU"]) for v in variacoes)
        + " MNDWI %.2f" % metricas(res["MNDWI"])["IoU"], flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa features VGG16 na entrada do SOM no Sen1Floods11")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--componentes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    parser.add_argument("--limite", type=int)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.pasta_cache, "features_cnn"), exist_ok=True)
    featuresCNN.carregar_pesos()  # Baixa os pesos antes de abrir os processos

    recortes = []
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            recortes.append(r)
    recortes = recortes[:args.limite] if args.limite else recortes
    print("%d recortes" % len(recortes), flush=True)
    with Pool(args.processos) as pool:
        resultados = [r for r in pool.map(rodar, [(r, args.componentes, args.sementes, args.pasta_cache)
                                                    for r in recortes], chunksize=1) if r]

    variacoes = ["SOM"] + ["SOM + VGG %d" % k for k in args.componentes]
    nomes = variacoes + ["NDWI", "MNDWI"]
    faixas = [("Todos", 0, 101), ("0-1%", 0, 1), ("1-5%", 1, 5), ("5-15%", 5, 15), ("15-40%", 15, 40), ("40-100%", 40, 101)]
    tabela = {}
    for nome_faixa, a, b in faixas:
        sel = [r for r in resultados if a <= r["pct_agua_rotulo"] < b]
        tabela[nome_faixa] = {"n": len(sel)}
        if not sel:
            continue
        for v in variacoes:
            por_semente = [metricas(somar([r["SOM"][v][s] for r in sel])) for s in range(args.sementes)]
            tabela[nome_faixa][v] = {k: float(np.mean([m[k] for m in por_semente])) for k in por_semente[0]}
        for m in ("NDWI", "MNDWI"):
            tabela[nome_faixa][m] = metricas(somar([r[m] for r in sel]))
    com_agua = [r for r in resultados if r["pct_agua_rotulo"] >= 1]
    def iou_recorte(r, n):
        return np.mean([metricas(c)["IoU"] for c in r["SOM"][n]]) if n in variacoes else metricas(r[n])["IoU"]
    falhas = {n: sum(1 for r in com_agua if iou_recorte(r, n) < 0.05) for n in nomes}
    # Recortes em que cada variação com VGG ganha ou perde do SOM sem features (diferença de IoU > 0,05)
    comparacao = {v: {"melhor": sum(1 for r in com_agua if iou_recorte(r, v) - iou_recorte(r, "SOM") > 0.05),
                      "pior": sum(1 for r in com_agua if iou_recorte(r, v) - iou_recorte(r, "SOM") < -0.05)}
                  for v in variacoes[1:]}
    tempo = {v: float(np.mean([t for r in resultados for t in r["tempo"][v]])) for v in variacoes}
    tempo_features = float(np.mean([r["tempo_features_s"] for r in resultados]))

    print("\n%d recortes avaliados" % len(resultados))
    print("%-9s %4s " % ("Água", "n") + " ".join("%12s" % n.replace("SOM + ", "+") for n in nomes) + "  (IoU da água)")
    for nome_faixa, _, _ in faixas:
        t = tabela[nome_faixa]
        print("%-9s %4d " % (nome_faixa, t["n"]) + " ".join("%12.3f" % t[n]["IoU"] if n in t else "%12s" % "-" for n in nomes))
    for chave in ("precisao", "sensibilidade"):
        print("%-14s " % chave + " ".join("%12.3f" % tabela["Todos"][n][chave] for n in nomes))
    print("Falhas (IoU < 0,05) em %d recortes com água: " % len(com_agua) + ", ".join("%s %d" % (n, v) for n, v in falhas.items()))
    print("Recortes em que a VGG muda a IoU em mais de 0,05: " + ", ".join(
        "%s: %d melhores, %d piores" % (v, c["melhor"], c["pior"]) for v, c in comparacao.items()))
    print("Tempo por recorte: features VGG + PCA %.1f s; SOM " % tempo_features
          + ", ".join("%s %.2f s" % (v, t) for v, t in tempo.items()))

    with open(os.path.join(args.saida, "features_cnn.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "sementes": args.sementes, "unidades": enchenteSOM.SOM_SIZE,
                   "metricas": tabela, "falhas_recortes_com_agua": falhas, "comparacao_com_som": comparacao,
                   "tempo_som_s": tempo, "tempo_features_s": tempo_features}, f, ensure_ascii=False, indent=1)

    cores = {"SOM": "#3fa9f5", "NDWI": "#e7298a", "MNDWI": "#ff9f1c"}
    tons = ["#1f5f99", "#0b2f52", "#5e3c99", "#2ca25f"]
    for k, v in enumerate(variacoes[1:]):
        cores[v] = tons[k % len(tons)]
    fig, ax = plt.subplots(1, 2, figsize=(20, 5.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    w = 0.8 / len(nomes)
    for k, n in enumerate(nomes):
        v = [tabela[f][n]["IoU"] if n in tabela[f] else 0 for f, _, _ in faixas]
        ax[0].bar(np.arange(len(faixas)) + (k - (len(nomes) - 1) / 2) * w, v, w, color=cores[n], label=n, edgecolor="k", lw=0.4)
    ax[0].set_xticks(np.arange(len(faixas)))
    ax[0].set_xticklabels(["%s\n(%d)" % (f, tabela[f]["n"]) for f, _, _ in faixas])
    ax[0].set_xlabel("Fração de água no recorte (número de recortes)")
    ax[0].set_ylabel("IoU da água (rótulo à mão)")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(ncol=len(nomes), fontsize=9)
    ax[0].set_title("IoU por fração de água")
    ax[1].bar(np.arange(len(nomes)), [falhas[n] for n in nomes], color=[cores[n] for n in nomes], edgecolor="k")
    ax[1].set_xticks(np.arange(len(nomes)))
    ax[1].set_xticklabels([n.replace("SOM + ", "SOM+\n") for n in nomes])
    ax[1].set_title("Recortes com água (n=%d) em que o método não achou água" % len(com_agua), fontsize=11)
    ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Sen1Floods11: features da VGG16 (ImageNet) na entrada do SOM (%d unidades, treino com 2%%)"
                 % enchenteSOM.SOM_SIZE, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "features_cnn.png"), dpi=85)
    print("Resultados salvos em", args.saida)
