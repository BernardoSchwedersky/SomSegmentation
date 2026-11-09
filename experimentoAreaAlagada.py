# -*- coding: utf-8 -*-
"""
Avalia a detecção só da área alagada no Sen1Floods11, com a abordagem
bi-temporal no SOM e no MNDWI: a imagem anterior define a água permanente, a
imagem do alagamento define a água total (alagamento + permanente) e o
alagamento é a diferença entre as duas.

- MNDWI (e NDWI): limiar 0 em cada data; alagamento = água só na imagem do
  alagamento (enchenteSOM.mapa_indice, classe 2).
- SOM bi-temporal: a mesma regra aplicada a cada unidade, pelo MNDWI médio dos
  seus pixels nas duas datas (classe 2). Variações: treinado em cada recorte, e
  pré-treinado com uma imagem balanceada de cada evento, juntas (todos os
  eventos ou sem o evento avaliado), normalização fixa e classe local.

Referência do alagamento: água no rótulo à mão do Sen1Floods11 (LabelHand) que
não é água permanente no JRC Global Surface Water (JRCWaterHand, distribuído com
o Sen1Floods11 na mesma grade). A água permanente prevista (classe 1) também é
comparada com o JRC.

Usa os recortes, as bandas e as imagens de pré-treino dos experimentos
anteriores (mesma pasta de cache) e avalia os mesmos 384 recortes de
experimentoPreTreinoEvento.py.

Uso:
    python experimentoAreaAlagada.py --sementes 3
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
import rasterio

import enchenteSOM
from avaliarSen1Floods11 import URL_BASE, baixar, contar, metricas, somar
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from experimentoPreTreinoSOM import PRE_TREINO, pre_treinar
from variacaoUnidadesSOM import ler_recorte

METODOS_SOM = ["SOM por recorte", "SOM pré-treinado (todos os eventos)", "SOM pré-treinado (outros eventos)"]
NOMES = METODOS_SOM + ["NDWI", "MNDWI"]


def ler_jrc(recorte, pasta_cache):
    arquivo = baixar("%s/data/flood_events/HandLabeled/JRCWaterHand/%s_JRCWaterHand.tif" % (URL_BASE, recorte),
                     os.path.join(pasta_cache, "rotulos", recorte + "_JRCWaterHand.tif"))
    with rasterio.open(arquivo) as src:
        return src.read(1) == 1

def avaliar(args):
    r, modelos, sementes, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "area_alagada", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    permanente_ref = ler_jrc(r["recorte"], pasta_cache)
    validos = (rotulo >= 0) & ~invalidos
    alagado_ref = (rotulo == 1) & ~permanente_ref
    res = {"recorte": r["recorte"], "pais": r["pais"], "pct_agua_rotulo": r["pct_agua_rotulo"],
           "pct_alagado_ref": round(100 * float(alagado_ref[validos].mean()), 2),
           "pct_permanente_ref": round(100 * float(permanente_ref[validos].mean()), 2),
           "alagamento": {}, "permanente": {}}

    def registrar(nome, mapa):
        res["alagamento"].setdefault(nome, []).append(contar(mapa == 2, alagado_ref, validos))
        res["permanente"].setdefault(nome, []).append(contar(mapa == 1, permanente_ref, validos))

    for ind in ("NDWI", "MNDWI"):
        registrar(ind, enchenteSOM.mapa_indice(antes, depois, 0, ind))
    entrada_percentil, _ = enchenteSOM.montar_entrada(antes, depois)
    entrada_fixa, _ = enchenteSOM.montar_entrada(antes, depois, normalizacao="fixa")
    for s in range(sementes):
        _, result = enchenteSOM.treinar_e_segmentar(entrada_percentil, invalidos, seed=s)
        classe, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, enchenteSOM.SOM_SIZE)
        registrar("SOM por recorte", classe[result])
    for nome, lista in modelos.items():
        for SOM, _ in lista:
            result = enchenteSOM.segmentar(SOM, entrada_fixa, 1)
            classe, _ = enchenteSOM.classificar_unidades(result, invalidos, antes, depois, enchenteSOM.SOM_SIZE)
            registrar(nome, classe[result])
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s alagado %5.1f%% | " % (r["recorte"], res["pct_alagado_ref"]) + " ".join(
        "%s %.2f" % (n.split(" (")[-1][:5] if "pré" in n else n[:5], metricas(somar(res["alagamento"][n]))["IoU"])
        for n in NOMES), flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia a detecção só da área alagada (bi-temporal) no Sen1Floods11")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.pasta_cache, "area_alagada"), exist_ok=True)

    por_evento = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    escolhidos = {p: escolher_pre_treino(l) for p, l in por_evento.items()}
    excluidos = set(r["recorte"] for r in escolhidos.values()) | set(PRE_TREINO)
    imagens = {p: e["recorte"] for p, e in escolhidos.items() if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX}

    todos = [pre_treinar(sorted(imagens.values()), "fixa", s, args.pasta_cache) for s in range(args.sementes)]
    tarefas = []
    for p, l in sorted(por_evento.items()):
        outros = sorted(r for q, r in imagens.items() if q != p)
        modelos = {"SOM pré-treinado (todos os eventos)": todos,
                   "SOM pré-treinado (outros eventos)": [pre_treinar(outros, "fixa", s, args.pasta_cache)
                                                          for s in range(args.sementes)]}
        tarefas += [(r, modelos, args.sementes, args.pasta_cache) for r in l if r["recorte"] not in excluidos]
        print("%-10s pré-treinos prontos" % p, flush=True)
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar, tarefas, chunksize=1)

    def tabela_de(sel, chave):
        t = {"n": len(sel)}
        for n in NOMES:
            k = len(sel[0][chave][n]) if sel else 0
            ms = [metricas(somar([r[chave][n][s] for r in sel])) for s in range(k)]
            t[n] = {m: float(np.mean([x[m] for x in ms])) for m in ms[0]} if ms else None
        return t

    faixas = [("Todos", 0, 101), ("Sem alagamento (<0,1%)", 0, 0.1), ("0,1-1%", 0.1, 1), ("1-5%", 1, 5),
              ("5-20%", 5, 20), ("20-100%", 20, 101)]
    tabela = {f: tabela_de([r for r in resultados if a <= r["pct_alagado_ref"] < b], "alagamento") for f, a, b in faixas}
    eventos = sorted(por_evento)
    tabela_eventos = {p: tabela_de([r for r in resultados if r["pais"] == p], "alagamento") for p in eventos}
    tabela_permanente = tabela_de(resultados, "permanente")
    com_alagamento = [r for r in resultados if r["pct_alagado_ref"] >= 1]
    falhas = {n: sum(1 for r in com_alagamento if np.mean([metricas(c)["IoU"] for c in r["alagamento"][n]]) < 0.05)
              for n in NOMES}
    # Pixels de alagamento previstos em recortes sem alagamento na referência (falsos alarmes)
    sem = [r for r in resultados if r["pct_alagado_ref"] < 0.1]
    falso_alarme = {n: float(np.mean([np.mean([c["fp"] / max(c["fp"] + c["vn"], 1) for c in r["alagamento"][n]])
                                      for r in sem])) * 100 for n in NOMES}

    print("\n%d recortes avaliados; %.1f%% dos pixels válidos são alagamento na referência" % (
        len(resultados), 100 * sum(somar(r["alagamento"]["MNDWI"])["vp"] + somar(r["alagamento"]["MNDWI"])["fn"]
                                   for r in resultados) / sum(sum(somar(r["alagamento"]["MNDWI"]).values())
                                                              for r in resultados)))
    print("\nALAGAMENTO (classe 2 x rótulo de água que não é permanente no JRC)")
    print("%-38s %6s %6s %6s %6s %6s | %6s %6s %6s" % ("Método", "IoU", "F1", "prec.", "sens.", "falhas",
                                                     "1-5%", "5-20%", ">20%"))
    for n in NOMES:
        t = tabela["Todos"][n]
        print("%-38s %6.3f %6.3f %6.3f %6.3f %6d | " % (n, t["IoU"], t["F1"], t["precisao"], t["sensibilidade"], falhas[n])
              + " ".join("%6.3f" % tabela[f][n]["IoU"] for f in ("1-5%", "5-20%", "20-100%")))
    print("Falso alarme em %d recortes sem alagamento (%% dos pixels marcados como alagamento): " % len(sem)
          + ", ".join("%s %.2f%%" % (n, v) for n, v in falso_alarme.items()))
    print("\nÁGUA PERMANENTE (classe 1 x JRC)")
    for n in NOMES:
        t = tabela_permanente[n]
        print("%-38s IoU %.3f  precisão %.3f  sensibilidade %.3f" % (n, t["IoU"], t["precisao"], t["sensibilidade"]))
    print("\nIoU do alagamento por evento:")
    print("%-10s %4s %9s " % ("Evento", "n", "alagado") + " ".join("%10s" % (n.split("(")[-1][:9] if "(" in n else n[:9])
                                                                  for n in NOMES))
    for p in eventos:
        sel = [r for r in resultados if r["pais"] == p]
        pct = np.mean([r["pct_alagado_ref"] for r in sel])
        print("%-10s %4d %8.1f%% " % (p, len(sel), pct) + " ".join("%10.3f" % tabela_eventos[p][n]["IoU"] for n in NOMES))

    # Diagnóstico com o MNDWI: onde vai cada pixel de alagamento da referência
    diag = defaultdict(lambda: np.zeros(5))
    for r in resultados:
        rr = next(x for x in por_evento[r["pais"]] if x["recorte"] == r["recorte"])
        rotulo, antes, depois, invalidos = ler_recorte(rr, args.pasta_cache)
        alagado = (rotulo == 1) & ~ler_jrc(r["recorte"], args.pasta_cache) & (rotulo >= 0) & ~invalidos
        agua_antes = enchenteSOM.indice_agua(antes) > 0
        agua_depois = enchenteSOM.indice_agua(depois) > 0
        dias = (np.datetime64(rr["data_depois"]) - np.datetime64(rr["data_antes"])).astype(int)
        for chave in (r["pais"], "Todos"):
            diag[chave] += [alagado.sum(), (alagado & agua_antes & agua_depois).sum(), (alagado & ~agua_depois).sum(),
                            (alagado & ~agua_antes & agua_depois).sum(), dias]
    n_eventos = {p: sum(1 for r in resultados if r["pais"] == p) for p in eventos}
    n_eventos["Todos"] = len(resultados)
    diagnostico = {p: {"ja_agua_na_imagem_anterior_pct": 100 * d[1] / max(d[0], 1),
                       "nao_agua_na_imagem_do_evento_pct": 100 * d[2] / max(d[0], 1),
                       "alagamento_detectado_pct": 100 * d[3] / max(d[0], 1),
                       "dias_entre_imagens_media": d[4] / n_eventos[p]} for p, d in diag.items()}
    print("\nDiagnóstico (MNDWI): destino dos pixels de alagamento da referência")
    print("%-10s %22s %22s %18s %8s" % ("Evento", "já água no antes", "não água no evento", "detectado", "dias"))
    for p in ["Todos"] + eventos:
        d = diagnostico[p]
        print("%-10s %21.0f%% %21.0f%% %17.0f%% %8.0f" % (p, d["ja_agua_na_imagem_anterior_pct"],
              d["nao_agua_na_imagem_do_evento_pct"], d["alagamento_detectado_pct"], d["dias_entre_imagens_media"]))

    with open(os.path.join(args.saida, "area_alagada.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "sementes": args.sementes, "unidades": enchenteSOM.SOM_SIZE,
                   "referencia": "LabelHand água e não JRCWaterHand", "alagamento": tabela,
                   "alagamento_por_evento": tabela_eventos, "permanente": tabela_permanente,
                   "falhas_recortes_com_alagamento": falhas, "recortes_com_alagamento": len(com_alagamento),
                   "falso_alarme_pct_sem_alagamento": falso_alarme, "recortes_sem_alagamento": len(sem),
                   "diagnostico_mndwi": diagnostico},
                  f, ensure_ascii=False, indent=1)

    # Figura: IoU do alagamento por evento e por fração de alagamento, e precisão/sensibilidade
    cores = {"SOM por recorte": "#3fa9f5", "SOM pré-treinado (todos os eventos)": "#6a51a3",
             "SOM pré-treinado (outros eventos)": "#bcbddc", "NDWI": "#e7298a", "MNDWI": "#ff9f1c"}
    fig, ax = plt.subplots(1, 3, figsize=(26, 6.5), gridspec_kw=dict(width_ratios=[2.2, 1, 0.9]))
    grupos = ["Todos"] + eventos
    w = 0.8 / len(NOMES)
    for k, n in enumerate(NOMES):
        v = [tabela["Todos"][n]["IoU"]] + [tabela_eventos[p][n]["IoU"] for p in eventos]
        ax[0].bar(np.arange(len(grupos)) + (k - (len(NOMES) - 1) / 2) * w, v, w, color=cores[n], label=n,
                  edgecolor="k", lw=0.4)
    ax[0].set_xticks(np.arange(len(grupos)))
    ax[0].set_xticklabels(["Todos\n(%d)" % len(resultados)] + ["%s\n(%d)" % (p, tabela_eventos[p]["n"]) for p in eventos])
    ax[0].set_ylabel("IoU do alagamento")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=9, ncol=3, loc="upper left")
    ax[0].set_title("Por evento (número de recortes)")
    nomes_faixa = ["1-5%", "5-20%", "20-100%"]
    for n in NOMES:
        ax[1].plot(nomes_faixa, [tabela[f][n]["IoU"] for f in nomes_faixa], "o-", lw=2, color=cores[n], label=n)
    ax[1].set_xticks(range(len(nomes_faixa)))
    ax[1].set_xticklabels(["%s\n(%d)" % (f, tabela[f]["n"]) for f in nomes_faixa])
    ax[1].set_xlabel("Fração de alagamento no recorte (número de recortes)")
    ax[1].set_ylim(0, 1)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Por fração de alagamento")
    for n in NOMES:
        t = tabela["Todos"][n]
        ax[2].scatter(t["sensibilidade"], t["precisao"], s=140, color=cores[n], edgecolor="k", label=n, zorder=3)
    ax[2].set_xlabel("Sensibilidade (alagamento encontrado)")
    ax[2].set_ylabel("Precisão (alagamento previsto que é real)")
    ax[2].set_xlim(0.3, 1)
    ax[2].set_ylim(0.3, 1)
    ax[2].grid(alpha=0.3)
    ax[2].set_title("Precisão x sensibilidade (todos os recortes)")
    fig.suptitle("Sen1Floods11: só a área alagada (água no evento que não é permanente no JRC) - SOM bi-temporal x "
                 "NDWI e MNDWI bi-temporais (%d unidades, %d sorteios)" % (enchenteSOM.SOM_SIZE, args.sementes), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "area_alagada.png"), dpi=85)
    print("Resultados salvos em", args.saida)
