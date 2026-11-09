# -*- coding: utf-8 -*-
"""
Testa a inclusão do radar Sentinel-1 (VV e VH da data do evento, distribuídos
pelo Sen1Floods11 na mesma grade dos rótulos, S1Hand) na detecção da área
alagada com o SOM bi-temporal.

O radar vê a água sob vegetação rala e com nuvens, que o MNDWI não vê: a água
aberta é escura no radar (VH baixo). Como o Planetary Computer não tem a cena de
radar de todas as datas de evento, o radar entra só na data do evento; a água
permanente continua vindo do MNDWI da imagem óptica anterior.

Regra bi-temporal (por pixel nos índices, por unidade no SOM):
    água antes  = MNDWI antes > 0
    água evento = MNDWI evento > 0 ou VH evento < limiar (dB)  [com radar]
                  (opcionalmente só onde MNDWI evento > mndwi_min: sem isso, solo
                  nu e areia secos, também escuros no VH, viram água)
    alagamento  = água no evento e não antes

Variações:
- MNDWI pixel a pixel, sem e com o radar;
- SOM óptico pré-treinado (uma imagem balanceada de cada evento, normalização
  fixa), classificando as unidades sem e com o radar;
- SOM com o radar também na entrada (VV e VH como canais extras), pré-treinado
  nas mesmas imagens.
Referência e recortes: os de experimentoAreaAlagada.py (rótulo à mão sem a água
permanente do JRC; 384 recortes). Pixels sem radar entram como inválidos em
todas as variações.

Uso:
    python experimentoRadar.py --limiares -20 -22 -24 --mndwi-min -0.1 -0.2 --sementes 3
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
from experimentoAreaAlagada import ler_jrc
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from experimentoPreTreinoSOM import PRE_TREINO
from variacaoUnidadesSOM import ler_recorte

# Faixas (dB) que viram 0-255 na entrada do SOM
FAIXA_VV = (-25.0, 0.0)
FAIXA_VH = (-32.0, -8.0)


def ler_radar(recorte, pasta_cache):
    arquivo = baixar("%s/data/flood_events/HandLabeled/S1Hand/%s_S1Hand.tif" % (URL_BASE, recorte),
                     os.path.join(pasta_cache, "radar", recorte + "_S1Hand.tif"))
    with rasterio.open(arquivo) as src:
        vv, vh = src.read(1).astype(float), src.read(2).astype(float)
    return vv, vh

def escalar(x, faixa):
    return np.clip((np.nan_to_num(x, nan=faixa[0]) - faixa[0]) / (faixa[1] - faixa[0]), 0, 1) * 255

# Entrada do SOM: bandas + MNDWI das duas datas (normalização fixa), mais VV e VH do evento se com_radar
def montar(antes, depois, vv, vh, com_radar):
    entrada, _ = enchenteSOM.montar_entrada(antes, depois, normalizacao="fixa")
    if com_radar:
        entrada = np.concatenate([entrada, escalar(vv, FAIXA_VV)[:, :, None], escalar(vh, FAIXA_VH)[:, :, None]], axis=2)
    return entrada

# Classe de cada unidade pela regra bi-temporal, com o VH médio da unidade se limiar_vh não for None
# (e, se mndwi_min não for None, só nas unidades com MNDWI médio do evento acima dele)
def classificar(result, invalidos, antes, depois, vh, som_size, limiar_vh=None, mndwi_min=None):
    m_antes, m_depois = enchenteSOM.indice_agua(antes), enchenteSOM.indice_agua(depois)
    classe = np.zeros(som_size, dtype=int)
    for u in range(som_size):
        m = (result == u) & ~invalidos
        if not np.any(m):
            continue
        md = m_depois[m].mean()
        agua_depois = md > 0 or (limiar_vh is not None and np.mean(vh[m]) < limiar_vh
                                 and (mndwi_min is None or md > mndwi_min))
        classe[u] = enchenteSOM.classe_por_agua(m_antes[m].mean() > 0, agua_depois)
    return classe

def mapa_pixel(antes, depois, vh, limiar_vh=None, mndwi_min=None):
    m_depois = enchenteSOM.indice_agua(depois)
    agua_depois = m_depois > 0
    if limiar_vh is not None:
        radar = np.nan_to_num(vh, nan=0) < limiar_vh
        if mndwi_min is not None:
            radar &= m_depois > mndwi_min
        agua_depois |= radar
    return enchenteSOM.classe_por_agua(enchenteSOM.indice_agua(antes) > 0, agua_depois)

def pre_treinar_radar(recortes, com_radar, seed, pasta_cache):
    rng = np.random.RandomState(seed)
    dados, amostras = [], []
    for r in recortes:
        rotulo, antes, depois, invalidos = ler_recorte({"recorte": r}, pasta_cache)
        vv, vh = ler_radar(r, pasta_cache)
        invalidos = invalidos | np.isnan(vh)
        entrada = montar(antes, depois, vv, vh, com_radar)
        validos = np.flatnonzero(~invalidos.ravel())
        sel = rng.choice(validos, int(round(len(validos) * 0.02)), replace=False)
        amostras.append(enchenteSOM.extrair_janelas(entrada, 1, sel))
    treino = np.concatenate(amostras)
    rng.shuffle(treino)
    enchenteSOM.somSeg.rand.seed(seed)
    SOM = enchenteSOM.somSeg.create_map(enchenteSOM.SOM_SIZE, 1, treino.shape[1])
    return enchenteSOM.somSeg.train_SOM(SOM, treino)

def avaliar(args):
    r, modelos, limiares, minimos, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "radar_resultados", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    vv, vh = ler_radar(r["recorte"], pasta_cache)
    invalidos = invalidos | np.isnan(vh)
    jrc = ler_jrc(r["recorte"], pasta_cache)
    validos = (rotulo >= 0) & ~invalidos
    ref = (rotulo == 1) & ~jrc
    res = {"recorte": r["recorte"], "pais": r["pais"], "pct_alagado_ref": round(100 * float(ref[validos].mean()), 2),
           "variacoes": {}}

    def registrar(nome, mapa):
        res["variacoes"].setdefault(nome, []).append(contar(mapa == 2, ref, validos))

    regras = [(t, None) for t in limiares] + [(t, c) for t in limiares for c in minimos]
    def nome_regra(t, c):
        return "VH < %g" % t + ("" if c is None else ", MNDWI > %g" % c)
    registrar("MNDWI", mapa_pixel(antes, depois, vh))
    for t, c in regras:
        registrar("MNDWI + radar (%s)" % nome_regra(t, c), mapa_pixel(antes, depois, vh, t, c))
    for com_radar, lista in modelos.items():
        entrada = montar(antes, depois, vv, vh, com_radar)
        base = "SOM com radar na entrada" if com_radar else "SOM óptico"
        for SOM in lista:
            result = enchenteSOM.segmentar(SOM, entrada, 1)
            registrar(base, classificar(result, invalidos, antes, depois, vh, enchenteSOM.SOM_SIZE)[result])
            for t, c in regras:
                registrar("%s + radar (%s)" % (base, nome_regra(t, c)),
                          classificar(result, invalidos, antes, depois, vh, enchenteSOM.SOM_SIZE, t, c)[result])
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s alagado %5.1f%%" % (r["recorte"], res["pct_alagado_ref"]), flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o radar Sentinel-1 na detecção do alagamento com o SOM")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--limiares", nargs="+", type=float, default=[-20, -22, -24])
    parser.add_argument("--mndwi-min", nargs="*", type=float, default=[-0.1, -0.2],
                        help="Restrições: radar só conta onde o MNDWI do evento é maior que esse valor")
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    for sub in ("radar", "radar_resultados"):
        os.makedirs(os.path.join(args.pasta_cache, sub), exist_ok=True)

    por_evento = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    escolhidos = {p: escolher_pre_treino(l) for p, l in por_evento.items()}
    excluidos = set(r["recorte"] for r in escolhidos.values()) | set(PRE_TREINO)
    imagens = sorted(e["recorte"] for e in escolhidos.values() if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX)
    modelos = {c: [pre_treinar_radar(imagens, c, s, args.pasta_cache) for s in range(args.sementes)] for c in (False, True)}
    print("Pré-treinos prontos (%d imagens)" % len(imagens), flush=True)

    tarefas = [(r, modelos, args.limiares, args.mndwi_min, args.pasta_cache) for p in sorted(por_evento) for r in por_evento[p]
               if r["recorte"] not in excluidos]
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar, tarefas, chunksize=1)

    nomes = list(resultados[0]["variacoes"])
    def tabela_de(sel):
        t = {"n": len(sel)}
        for n in nomes:
            k = len(sel[0]["variacoes"][n])
            ms = [metricas(somar([r["variacoes"][n][s] for r in sel])) for s in range(k)]
            t[n] = {m: float(np.mean([x[m] for x in ms])) for m in ms[0]}
        return t
    tabela = tabela_de(resultados)
    eventos = sorted(por_evento)
    tabela_eventos = {p: tabela_de([r for r in resultados if r["pais"] == p]) for p in eventos}
    com_alagamento = [r for r in resultados if r["pct_alagado_ref"] >= 1]
    falhas = {n: sum(1 for r in com_alagamento if np.mean([metricas(c)["IoU"] for c in r["variacoes"][n]]) < 0.05)
              for n in nomes}

    print("\n%d recortes avaliados - ALAGAMENTO (rótulo à mão sem a água permanente do JRC)" % len(resultados))
    print("%-62s %6s %6s %6s %6s %7s %9s" % ("Variação", "IoU", "F1", "prec.", "sens.", "falhas", "Paraguai"))
    for n in nomes:
        t = tabela[n]
        print("%-62s %6.3f %6.3f %6.3f %6.3f %7d %9.3f" % (n, t["IoU"], t["F1"], t["precisao"], t["sensibilidade"],
                                                          falhas[n], tabela_eventos["Paraguay"][n]["IoU"]))
    with open(os.path.join(args.saida, "radar.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "sementes": args.sementes, "limiares_vh_db": args.limiares,
                   "mndwi_min": args.mndwi_min,
                   "imagens_pre_treino": imagens, "alagamento": tabela, "alagamento_por_evento": tabela_eventos,
                   "falhas_recortes_com_alagamento": falhas, "recortes_com_alagamento": len(com_alagamento)},
                  f, ensure_ascii=False, indent=1)

    # Figura: IoU por evento das variações principais (melhor regra do radar com restrição no MNDWI, pela IoU)
    melhor = max([n for n in nomes if n.startswith("MNDWI + radar") and "MNDWI >" in n], key=lambda n: tabela[n]["IoU"])
    regra = melhor[len("MNDWI + radar ("):-1]
    principais = ["MNDWI", melhor, "SOM óptico", "SOM óptico + radar (%s)" % regra,
                  "SOM com radar na entrada + radar (%s)" % regra]
    cores = ["#ff9f1c", "#b35806", "#bcbddc", "#6a51a3", "#2d004b"]
    fig, ax = plt.subplots(1, 2, figsize=(24, 6.5), gridspec_kw=dict(width_ratios=[2.3, 1]))
    grupos = ["Todos"] + eventos
    w = 0.8 / len(principais)
    for k, n in enumerate(principais):
        v = [tabela[n]["IoU"]] + [tabela_eventos[p][n]["IoU"] for p in eventos]
        ax[0].bar(np.arange(len(grupos)) + (k - (len(principais) - 1) / 2) * w, v, w, color=cores[k], label=n,
                  edgecolor="k", lw=0.4)
    ax[0].set_xticks(np.arange(len(grupos)))
    ax[0].set_xticklabels(["Todos\n(%d)" % len(resultados)] + ["%s\n(%d)" % (p, tabela_eventos[p]["n"]) for p in eventos])
    ax[0].set_ylabel("IoU do alagamento")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=9, ncol=2, loc="upper left")
    ax[0].set_title("Por evento (regra do radar: %s)" % regra)
    for k, n in enumerate(principais):
        ax[1].scatter(tabela[n]["sensibilidade"], tabela[n]["precisao"], s=150, color=cores[k], edgecolor="k", label=n)
    for n in nomes:  # Demais regras como pontos pequenos
        if n not in principais and n.startswith(("MNDWI + radar", "SOM óptico + radar")):
            c = "#b35806" if n.startswith("MNDWI") else "#6a51a3"
            ax[1].scatter(tabela[n]["sensibilidade"], tabela[n]["precisao"], s=25, color=c, alpha=0.6)
    ax[1].set_xlabel("Sensibilidade")
    ax[1].set_ylabel("Precisão")
    ax[1].set_xlim(0.2, 1)
    ax[1].set_ylim(0.2, 1)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Precisão x sensibilidade (pontos pequenos: outras regras do radar)")
    fig.suptitle("Sen1Floods11: radar Sentinel-1 do evento na detecção do alagamento (%d recortes; SOM com %d unidades, "
                 "%d sorteios)" % (len(resultados), enchenteSOM.SOM_SIZE, args.sementes), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "radar.png"), dpi=85)
    print("Resultados salvos em", args.saida)
