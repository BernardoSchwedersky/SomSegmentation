# -*- coding: utf-8 -*-
"""
Testa o limiar automático de Otsu para o índice de água (MNDWI), no MNDWI
direto e na classificação das unidades do SOM bi-temporal, no Sen1Floods11.

Para cada recorte e cada data (antes e evento), o limiar é calculado pelo método
de Otsu sobre o histograma do MNDWI dos pixels válidos. Variações do limiar:
- "fixo 0": o atual (Xu, 2006);
- "Otsu": o limiar de Otsu de cada imagem;
- "Otsu limitado": o limiar de Otsu restrito a [LIMITE_MIN, LIMITE_MAX], para
  imagens com pouca ou nenhuma água, em que o histograma não tem dois picos e o
  Otsu cai dentro da distribuição da terra.
O limiar de cada data vale para os pixels (MNDWI) e para o MNDWI médio de cada
unidade (SOM pré-treinado com uma imagem balanceada de cada evento, normalização
fixa, 36 unidades).

Avaliação nos mesmos recortes de experimentoAreaAlagada.py:
- alagamento: classe 2 contra o rótulo à mão sem a água permanente do JRC;
- água total: classes 1 e 2 contra o rótulo à mão.

Uso:
    python experimentoOtsu.py --sementes 3
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
from experimentoAreaAlagada import ler_jrc
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from experimentoPreTreinoSOM import PRE_TREINO, pre_treinar
from variacaoUnidadesSOM import ler_recorte

LIMITE_MIN, LIMITE_MAX = -0.2, 0.2
LIMIARES = ["fixo 0", "Otsu", "Otsu limitado"]


# Limiar de Otsu: maximiza a variância entre as duas classes do histograma
def otsu(valores, bins=256, faixa=(-1, 1)):
    hist, bordas = np.histogram(valores, bins=bins, range=faixa)
    centros = (bordas[:-1] + bordas[1:]) / 2
    p = hist / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centros)
    mu_t = mu[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    var[~np.isfinite(var)] = -1
    return float(centros[np.argmax(var)])

def limiares(indice, validos):
    t = otsu(indice[validos]) if validos.any() else 0.0
    return {"fixo 0": 0.0, "Otsu": t, "Otsu limitado": float(np.clip(t, LIMITE_MIN, LIMITE_MAX))}

def classificar(result, invalidos, m_antes, m_depois, som_size, t_antes, t_depois):
    classe = np.zeros(som_size, dtype=int)
    for u in range(som_size):
        m = (result == u) & ~invalidos
        if np.any(m):
            classe[u] = enchenteSOM.classe_por_agua(m_antes[m].mean() > t_antes, m_depois[m].mean() > t_depois)
    return classe

def avaliar(args):
    r, modelos, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "otsu", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    jrc = ler_jrc(r["recorte"], pasta_cache)
    validos = (rotulo >= 0) & ~invalidos
    ref_alag = (rotulo == 1) & ~jrc
    ref_agua = rotulo == 1
    m_antes, m_depois = enchenteSOM.indice_agua(antes), enchenteSOM.indice_agua(depois)
    lim_antes, lim_depois = limiares(m_antes, ~invalidos), limiares(m_depois, ~invalidos)
    res = {"recorte": r["recorte"], "pais": r["pais"], "pct_alagado_ref": round(100 * float(ref_alag[validos].mean()), 2),
           "pct_agua_rotulo": r["pct_agua_rotulo"], "limiar_antes": lim_antes, "limiar_depois": lim_depois,
           "alagamento": {}, "agua": {}}

    def registrar(nome, mapa):
        res["alagamento"].setdefault(nome, []).append(contar(mapa == 2, ref_alag, validos))
        res["agua"].setdefault(nome, []).append(contar(mapa >= 1, ref_agua, validos))

    for l in LIMIARES:
        registrar("MNDWI | " + l, enchenteSOM.classe_por_agua(m_antes > lim_antes[l], m_depois > lim_depois[l]))
    entrada, _ = enchenteSOM.montar_entrada(antes, depois, normalizacao="fixa")
    for SOM, _ in modelos:
        result = enchenteSOM.segmentar(SOM, entrada, 1)
        for l in LIMIARES:
            classe = classificar(result, invalidos, m_antes, m_depois, enchenteSOM.SOM_SIZE, lim_antes[l], lim_depois[l])
            registrar("SOM | " + l, classe[result])
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s Otsu antes %+.2f evento %+.2f" % (r["recorte"], lim_antes["Otsu"], lim_depois["Otsu"]), flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o limiar de Otsu no MNDWI e no SOM")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.pasta_cache, "otsu"), exist_ok=True)

    por_evento = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    escolhidos = {p: escolher_pre_treino(l) for p, l in por_evento.items()}
    excluidos = set(r["recorte"] for r in escolhidos.values()) | set(PRE_TREINO)
    imagens = sorted(e["recorte"] for e in escolhidos.values() if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX)
    modelos = [pre_treinar(imagens, "fixa", s, args.pasta_cache) for s in range(args.sementes)]
    tarefas = [(r, modelos, args.pasta_cache) for p in sorted(por_evento) for r in por_evento[p]
               if r["recorte"] not in excluidos]
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar, tarefas, chunksize=1)

    nomes = ["%s | %s" % (m, l) for m in ("MNDWI", "SOM") for l in LIMIARES]
    def tabela_de(sel, chave):
        t = {"n": len(sel)}
        for n in nomes:
            k = len(sel[0][chave][n]) if sel else 0
            ms = [metricas(somar([r[chave][n][s] for r in sel])) for s in range(k)]
            t[n] = {m: float(np.mean([x[m] for x in ms])) for m in ms[0]} if ms else None
        return t
    faixas = [("Todos", 0, 101), ("Sem alagamento (<0,1%)", 0, 0.1), ("0,1-1%", 0.1, 1), ("1-5%", 1, 5),
              ("5-20%", 5, 20), ("20-100%", 20, 101)]
    tab_alag = {f: tabela_de([r for r in resultados if a <= r["pct_alagado_ref"] < b], "alagamento") for f, a, b in faixas}
    tab_agua = tabela_de(resultados, "agua")
    eventos = sorted(por_evento)
    tab_eventos = {p: tabela_de([r for r in resultados if r["pais"] == p], "alagamento") for p in eventos}
    com_alag = [r for r in resultados if r["pct_alagado_ref"] >= 1]
    falhas = {n: sum(1 for r in com_alag if np.mean([metricas(c)["IoU"] for c in r["alagamento"][n]]) < 0.05) for n in nomes}
    sem = [r for r in resultados if r["pct_alagado_ref"] < 0.1]
    falso = {n: 100 * float(np.mean([np.mean([c["fp"] / max(c["fp"] + c["vn"], 1) for c in r["alagamento"][n]]) for r in sem]))
             for n in nomes}
    otsus = np.array([[r["limiar_antes"]["Otsu"], r["limiar_depois"]["Otsu"]] for r in resultados])

    print("\n%d recortes avaliados" % len(resultados))
    print("Limiar de Otsu do MNDWI: antes mediana %+.2f (p10 %+.2f, p90 %+.2f); evento mediana %+.2f (p10 %+.2f, p90 %+.2f)"
          % (np.median(otsus[:, 0]), *np.percentile(otsus[:, 0], [10, 90]), np.median(otsus[:, 1]),
             *np.percentile(otsus[:, 1], [10, 90])))
    print("\n%-26s %s" % ("", "ALAGAMENTO" + " " * 44 + "| ÁGUA TOTAL"))
    print("%-26s %6s %6s %6s %6s %7s %9s | %6s %6s %6s" % ("Método | limiar", "IoU", "prec.", "sens.", "1-5%",
                                                          "falhas", "f.alarme", "IoU", "prec.", "sens."))
    for n in nomes:
        t, a = tab_alag["Todos"][n], tab_agua[n]
        print("%-26s %6.3f %6.3f %6.3f %6.3f %7d %8.2f%% | %6.3f %6.3f %6.3f" % (
            n, t["IoU"], t["precisao"], t["sensibilidade"], tab_alag["1-5%"][n]["IoU"], falhas[n], falso[n],
            a["IoU"], a["precisao"], a["sensibilidade"]))
    print("\nIoU do alagamento por evento:")
    print("%-10s %4s " % ("Evento", "n") + " ".join("%13s" % n.replace(" | ", "/")[:13] for n in nomes))
    for p in eventos:
        print("%-10s %4d " % (p, tab_eventos[p]["n"]) + " ".join("%13.3f" % tab_eventos[p][n]["IoU"] for n in nomes))

    with open(os.path.join(args.saida, "otsu.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "sementes": args.sementes, "limites_otsu": [LIMITE_MIN, LIMITE_MAX],
                   "otsu_antes": {"mediana": float(np.median(otsus[:, 0]))},
                   "otsu_evento": {"mediana": float(np.median(otsus[:, 1]))},
                   "alagamento": tab_alag, "alagamento_por_evento": tab_eventos, "agua_total": tab_agua,
                   "falhas_recortes_com_alagamento": falhas, "falso_alarme_pct_sem_alagamento": falso},
                  f, ensure_ascii=False, indent=1)

    cores = {"fixo 0": "#ff9f1c", "Otsu": "#e7298a", "Otsu limitado": "#2ca25f"}
    fig, ax = plt.subplots(1, 3, figsize=(26, 6.2), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    grupos = ["Todos"] + eventos
    w = 0.8 / len(nomes)
    for k, n in enumerate(nomes):
        m, l = n.split(" | ")
        v = [tab_alag["Todos"][n]["IoU"]] + [tab_eventos[p][n]["IoU"] for p in eventos]
        ax[0].bar(np.arange(len(grupos)) + (k - (len(nomes) - 1) / 2) * w, v, w, color=cores[l], label=n,
                  edgecolor="k", lw=0.4, hatch="//" if m == "SOM" else None)
    ax[0].set_xticks(np.arange(len(grupos)))
    ax[0].set_xticklabels(["Todos\n(%d)" % len(resultados)] + ["%s\n(%d)" % (p, tab_eventos[p]["n"]) for p in eventos])
    ax[0].set_ylabel("IoU do alagamento")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    ax[0].set_title("Por evento (SOM com hachura)")
    ax[1].hist(otsus[:, 0], bins=40, range=(-1, 1), alpha=0.6, label="antes", color="#7f7f7f")
    ax[1].hist(otsus[:, 1], bins=40, range=(-1, 1), alpha=0.6, label="evento", color="#3fa9f5")
    for x in (LIMITE_MIN, 0, LIMITE_MAX):
        ax[1].axvline(x, color="k", ls=":" if x else "-", lw=1)
    ax[1].set_xlabel("Limiar de Otsu do MNDWI")
    ax[1].set_ylabel("Recortes")
    ax[1].legend()
    ax[1].set_title("Limiares de Otsu (pontilhado: limites)")
    for n in nomes:
        m, l = n.split(" | ")
        t = tab_alag["Todos"][n]
        ax[2].scatter(t["sensibilidade"], t["precisao"], s=150, color=cores[l], marker="s" if m == "SOM" else "o",
                      edgecolor="k", label=n)
    ax[2].set_xlabel("Sensibilidade do alagamento")
    ax[2].set_ylabel("Precisão do alagamento")
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=8)
    ax[2].set_title("Precisão x sensibilidade (quadrado: SOM)")
    fig.suptitle("Sen1Floods11: limiar do MNDWI fixo em 0 x Otsu por imagem, no MNDWI direto e no SOM pré-treinado "
                 "(%d recortes, área alagada)" % len(resultados), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "otsu.png"), dpi=85)
    print("Resultados salvos em", args.saida)
