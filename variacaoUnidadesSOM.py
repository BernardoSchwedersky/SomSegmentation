# -*- coding: utf-8 -*-
"""
Testa o número de unidades do SOM bi-temporal contra os rótulos à mão do
Sen1Floods11, usando os recortes e as cenas já escolhidos por
avaliarSen1Floods11.py (que precisa ter rodado antes, com a mesma pasta de cache).

Para cada recorte avaliado com sucesso, lê as bandas das duas cenas na grade do
rótulo (guardadas em <pasta_cache>/bandas/ para as próximas execuções), roda o
SOM com cada número de unidades e conta acertos e erros da água (água permanente
ou alagamento) contra o rótulo. O MNDWI e o NDWI diretos entram como referência.

Saída: resultados/sen1floods11/unidades_som.json e unidades_som.png.

Uso:
    python variacaoUnidadesSOM.py --unidades 4 8 12 24 36
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
import rasterio

import enchenteSOM
import sentinelUtils
from avaliarSen1Floods11 import contar, metricas, somar


# Bandas das duas cenas e máscara de inválidos na grade do rótulo (com cache)
def ler_recorte(r, pasta_cache):
    arquivo = os.path.join(pasta_cache, "bandas", r["recorte"] + ".npz")
    with rasterio.open(os.path.join(pasta_cache, "rotulos", r["recorte"] + "_LabelHand.tif")) as src:
        rotulo = src.read(1)
        grade = (src.crs, src.transform, src.width, src.height)
    if os.path.exists(arquivo):
        d = np.load(arquivo)
        return rotulo, d["antes"] / 10000, d["depois"] / 10000, d["invalidos"]
    item_antes, item_depois = sentinelUtils.buscar_item(r["cena_antes"]), sentinelUtils.buscar_item(r["cena_depois"])
    antes = sentinelUtils.ler_bandas_na_grade(item_antes, enchenteSOM.BANDAS, *grade)
    depois = sentinelUtils.ler_bandas_na_grade(item_depois, enchenteSOM.BANDAS, *grade)
    invalidos = (sentinelUtils.ler_mascara_nuvens_na_grade(item_antes, *grade)
                 | sentinelUtils.ler_mascara_nuvens_na_grade(item_depois, *grade))
    np.savez_compressed(arquivo, antes=np.round(antes * 10000).astype(np.uint16),
                        depois=np.round(depois * 10000).astype(np.uint16), invalidos=invalidos)
    return rotulo, np.round(antes * 10000) / 10000, np.round(depois * 10000) / 10000, invalidos

def avaliar(args):
    r, unidades, pasta_cache = args
    arquivo_resultado = os.path.join(pasta_cache, "unidades", r["recorte"] + ".json")
    feito = {}
    if os.path.exists(arquivo_resultado):
        with open(arquivo_resultado) as f:
            feito = json.load(f)
        if all(str(u) in feito["SOM"] for u in unidades):
            return feito
    try:
        rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    except Exception as erro:
        print("%-22s erro: %s" % (r["recorte"], erro), flush=True)
        return None
    validos = (rotulo >= 0) & ~invalidos
    referencia = rotulo == 1
    res = feito or {"recorte": r["recorte"], "pais": r["pais"], "pct_agua_rotulo": r["pct_agua_rotulo"], "SOM": {},
                    "NDWI": contar(np.isin(enchenteSOM.mapa_indice(antes, depois, 0, "NDWI"), [1, 2]), referencia, validos),
                    "MNDWI": contar(np.isin(enchenteSOM.mapa_indice(antes, depois, 0, "MNDWI"), [1, 2]), referencia, validos)}
    entrada, _ = enchenteSOM.montar_entrada(antes, depois)
    for u in unidades:
        if str(u) in res["SOM"]:
            continue
        _, result = enchenteSOM.treinar_e_segmentar(entrada, invalidos, som_size=u)
        classe_unidade, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, u)
        res["SOM"][str(u)] = contar(np.isin(classe_unidade[result], [1, 2]), referencia, validos)
    with open(arquivo_resultado, "w") as f:
        json.dump(res, f)
    print("%-22s " % r["recorte"] + " ".join("SOM%s %.2f" % (u, metricas(res["SOM"][str(u)])["IoU"]) for u in unidades)
          + " MNDWI %.2f" % metricas(res["MNDWI"])["IoU"], flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o número de unidades do SOM no Sen1Floods11")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--unidades", nargs="+", type=int, default=[4, 8, 12, 24, 36])
    parser.add_argument("--processos", type=int, default=4)
    parser.add_argument("--limite", type=int, help="Usa só os primeiros N recortes (teste rápido)")
    args = parser.parse_args()
    for sub in ("bandas", "unidades"):
        os.makedirs(os.path.join(args.pasta_cache, sub), exist_ok=True)

    recortes = []
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            recortes.append(r)
    recortes = recortes[:args.limite] if args.limite else recortes
    print("%d recortes, unidades %s" % (len(recortes), args.unidades), flush=True)
    with Pool(args.processos) as pool:
        resultados = [r for r in pool.map(avaliar, [(r, args.unidades, args.pasta_cache) for r in recortes], chunksize=1) if r]

    faixas = [("Todos", 0, 101), ("0-1%", 0, 1), ("1-5%", 1, 5), ("5-15%", 5, 15), ("15-40%", 15, 40), ("40-100%", 40, 101)]
    nomes = ["SOM %d" % u for u in args.unidades] + ["NDWI", "MNDWI"]
    tabela = {}
    for nome_faixa, a, b in faixas:
        sel = [r for r in resultados if a <= r["pct_agua_rotulo"] < b]
        tabela[nome_faixa] = {"n": len(sel)}
        for u in args.unidades:
            tabela[nome_faixa]["SOM %d" % u] = metricas(somar([r["SOM"][str(u)] for r in sel])) if sel else None
        for m in ("NDWI", "MNDWI"):
            tabela[nome_faixa][m] = metricas(somar([r[m] for r in sel])) if sel else None
    # Recortes com água em que o método praticamente não encontrou água (IoU < 0,05)
    com_agua = [r for r in resultados if r["pct_agua_rotulo"] >= 1]
    falhas = {n: sum(1 for r in com_agua if metricas(r["SOM"][n.split()[1]] if n.startswith("SOM") else r[n])["IoU"] < 0.05)
              for n in nomes}

    print("\n%d recortes avaliados" % len(resultados))
    print("%-9s %4s " % ("Água", "n") + " ".join("%8s" % n for n in nomes) + "   (IoU da água)")
    for nome_faixa, _, _ in faixas:
        t = tabela[nome_faixa]
        print("%-9s %4d " % (nome_faixa, t["n"]) + " ".join("%8.3f" % t[n]["IoU"] if t[n] else "%8s" % "-" for n in nomes))
    print("Falhas (IoU < 0,05) em %d recortes com >= 1%% de água: " % len(com_agua)
          + ", ".join("%s %d" % (n, v) for n, v in falhas.items()))

    with open(os.path.join(args.saida, "unidades_som.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "unidades": args.unidades, "metricas": tabela,
                   "falhas_recortes_com_agua": falhas, "recortes_com_agua": len(com_agua)}, f, ensure_ascii=False, indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(21, 5.5))
    x = np.arange(len(args.unidades))
    for chave, rotulo_m, estilo in (("IoU", "IoU", "o-"), ("precisao", "Precisão", "s--"), ("sensibilidade", "Sensibilidade", "^:")):
        ax[0].plot(x, [tabela["Todos"]["SOM %d" % u][chave] for u in args.unidades], estilo, color="#3fa9f5", lw=2.5,
                   label="SOM - " + rotulo_m)
        ax[0].axhline(tabela["Todos"]["MNDWI"][chave], color="#ff9f1c", ls=estilo[1:], lw=1.5, label="MNDWI - " + rotulo_m)
    ax[0].axhline(tabela["Todos"]["NDWI"]["IoU"], color="#e7298a", lw=1.5, label="NDWI - IoU")
    ax[0].set_title("Todos os recortes (%d)" % tabela["Todos"]["n"])
    for nome_faixa, _, _ in faixas[1:]:
        ax[1].plot(x, [tabela[nome_faixa]["SOM %d" % u]["IoU"] for u in args.unidades], "o-", lw=2,
                   label="%s de água (%d)" % (nome_faixa, tabela[nome_faixa]["n"]))
    ax[1].set_title("IoU do SOM por fração de água no recorte")
    ax[2].bar(np.arange(len(nomes)), [falhas[n] for n in nomes],
              color=["#3fa9f5"] * len(args.unidades) + ["#e7298a", "#ff9f1c"], edgecolor="k")
    ax[2].set_xticks(np.arange(len(nomes)))
    ax[2].set_xticklabels(nomes)
    ax[2].set_ylabel("Recortes")
    ax[2].set_title("Recortes com água (>= 1%%, n=%d) em que o método não achou água (IoU < 0,05)" % len(com_agua),
                    fontsize=10)
    for a in ax[:2]:
        a.set_xticks(x)
        a.set_xticklabels([str(u) for u in args.unidades])
        a.set_xlabel("Unidades do SOM")
        a.set_ylim(0, 1)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Sen1Floods11: efeito do número de unidades do SOM bi-temporal (água = permanente + alagamento)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "unidades_som.png"), dpi=85)
    print("Resultados salvos em", args.saida)
