# -*- coding: utf-8 -*-
"""
Testa o par de radar Sentinel-1 antes/depois (coleção sentinel-1-rtc do
Planetary Computer, retroespalhamento corrigido do terreno) na detecção da área
alagada, junto com o SOM bi-temporal e o MNDWI.

Escolha das cenas de radar, para cada recorte do Sen1Floods11:
- depois: a cena mais próxima da data do evento óptico, de 2 dias antes a 6 dias
  depois;
- antes: a cena mais recente da mesma órbita relativa (mesma geometria) até a
  data da imagem óptica de antes (ou, se não houver em 48 dias, até 10 dias antes
  da cena de depois).
As duas são lidas na grade do rótulo, com um filtro de média 5x5 (no linear)
contra o speckle, e convertidas para dB.

Regras de água no evento (a água permanente continua vindo do MNDWI de antes):
- "radar bi-temporal": água em cada data = VH < limiar; alagamento = água só
  depois (só radar, sem óptico);
- "mudança no radar": VH depois < limiar e queda do VH (depois - antes) maior
  que delta dB — a superfície escureceu, o que exclui solo e areia secos, escuros
  nas duas datas;
- óptico + mudança no radar: água no evento = MNDWI > 0 ou mudança no radar.
Nos SOMs (óptico pré-treinado, e com VV/VH de antes e depois na entrada), a
regra é aplicada à média de cada unidade.

Referência e recortes: os de experimentoAreaAlagada.py (rótulo à mão sem a água
permanente do JRC), sem os de pré-treino.

Uso:
    python experimentoRadarBitemporal.py --sementes 3
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pystac_client
import planetary_computer
import rasterio

import enchenteSOM
import sentinelUtils
from avaliarSen1Floods11 import contar, metricas, somar
from experimentoAreaAlagada import ler_jrc
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from experimentoPreTreinoSOM import PRE_TREINO
from variacaoUnidadesSOM import ler_recorte

LIMIAR_VH = -20.0   # dB: abaixo disso, água aberta
DELTA_VH = -3.0     # dB: queda mínima do VH entre as datas para contar como água nova
FAIXA_VV, FAIXA_VH = (-25.0, 0.0), (-32.0, -8.0)  # dB -> 0-255 na entrada do SOM


def media_movel(x, k=5):
    pad = np.pad(x, k // 2, mode="edge")
    c = np.cumsum(np.cumsum(pad, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)

def buscar_rtc(bbox, inicio, fim):
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    busca = cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime="%s/%s" % (inicio, fim))
    return sentinelUtils._repetir(lambda: list(busca.items()))

# Par de radar (dB) na grade do rótulo, com cache. Retorna dict com vv/vh antes e depois, datas e órbita
def par_radar(r, pasta_cache):
    arquivo = os.path.join(pasta_cache, "radar_par", r["recorte"] + ".npz")
    if os.path.exists(arquivo):
        d = np.load(arquivo, allow_pickle=True)
        return {k: d[k] for k in d.files} if d["ok"] else None
    with rasterio.open(os.path.join(pasta_cache, "rotulos", r["recorte"] + "_LabelHand.tif")) as src:
        grade = (src.crs, src.transform, src.width, src.height)
        bbox = list(src.bounds)
    evento = date.fromisoformat(r["data_depois"])
    candidatas = buscar_rtc(bbox, evento - timedelta(days=2), evento + timedelta(days=7))
    if not candidatas:
        np.savez(arquivo, ok=False)
        return None
    depois = min(candidatas, key=lambda i: abs((i.datetime.date() - evento).days))
    orbita = depois.properties["sat:relative_orbit"]
    data_antes_optica = date.fromisoformat(r["data_antes"])
    mesma = [i for i in buscar_rtc(bbox, data_antes_optica - timedelta(days=60), depois.datetime.date() - timedelta(days=1))
             if i.properties["sat:relative_orbit"] == orbita]
    antes = [i for i in mesma if data_antes_optica - timedelta(days=48) <= i.datetime.date() <= data_antes_optica]
    if not antes:
        antes = [i for i in mesma if i.datetime.date() <= depois.datetime.date() - timedelta(days=10)]
    if not antes:
        np.savez(arquivo, ok=False)
        return None
    antes = max(antes, key=lambda i: i.datetime)
    res = {"ok": True, "data_antes_radar": antes.datetime.date().isoformat(),
           "data_depois_radar": depois.datetime.date().isoformat(), "orbita": orbita}
    for nome, item in (("antes", antes), ("depois", depois)):
        for pol in ("vv", "vh"):
            lin = sentinelUtils.ler_asset_na_grade(item, pol, *grade)
            invalido = ~np.isfinite(lin) | (lin <= 0)
            lin = media_movel(np.where(invalido, np.nanmedian(np.where(invalido, np.nan, lin)), lin))
            db = 10 * np.log10(np.maximum(lin, 1e-6))
            db[invalido] = np.nan
            res["%s_%s" % (pol, nome)] = db.astype(np.float32)
    np.savez_compressed(arquivo, **res)
    return res

def mudanca_radar(rad):
    return (np.nan_to_num(rad["vh_depois"], nan=0) < LIMIAR_VH) & (
        np.nan_to_num(rad["vh_depois"] - rad["vh_antes"], nan=0) < DELTA_VH)

def mapas_pixel(antes, depois, rad):
    agua_antes = enchenteSOM.indice_agua(antes) > 0
    mndwi_depois = enchenteSOM.indice_agua(depois) > 0
    vh_a, vh_d = (np.nan_to_num(rad[k], nan=0) for k in ("vh_antes", "vh_depois"))
    return {"MNDWI": enchenteSOM.classe_por_agua(agua_antes, mndwi_depois),
            "Radar bi-temporal (VH < %g em cada data)" % LIMIAR_VH:
                enchenteSOM.classe_por_agua(vh_a < LIMIAR_VH, vh_d < LIMIAR_VH),
            "Mudança no radar (só radar)": np.where(mudanca_radar(rad), 2, 0),
            "MNDWI + mudança no radar": enchenteSOM.classe_por_agua(agua_antes, mndwi_depois | mudanca_radar(rad))}

# Classe por unidade com a regra óptico + mudança no radar (médias da unidade)
def classificar_unidades(result, invalidos, antes, depois, rad, som_size, com_radar):
    m_a, m_d = enchenteSOM.indice_agua(antes), enchenteSOM.indice_agua(depois)
    vh_a, vh_d = rad["vh_antes"], rad["vh_depois"]
    classe = np.zeros(som_size, dtype=int)
    for u in range(som_size):
        m = (result == u) & ~invalidos
        if not np.any(m):
            continue
        agua_d = m_d[m].mean() > 0
        if com_radar:
            agua_d = agua_d or (np.mean(vh_d[m]) < LIMIAR_VH and np.mean(vh_d[m] - vh_a[m]) < DELTA_VH)
        classe[u] = enchenteSOM.classe_por_agua(m_a[m].mean() > 0, agua_d)
    return classe

def escalar(x, faixa):
    return np.clip((np.nan_to_num(x, nan=faixa[0]) - faixa[0]) / (faixa[1] - faixa[0]), 0, 1) * 255

def montar(antes, depois, rad, com_radar):
    entrada, _ = enchenteSOM.montar_entrada(antes, depois, normalizacao="fixa")
    if com_radar:
        extras = [escalar(rad["vv_antes"], FAIXA_VV), escalar(rad["vh_antes"], FAIXA_VH),
                  escalar(rad["vv_depois"], FAIXA_VV), escalar(rad["vh_depois"], FAIXA_VH)]
        entrada = np.concatenate([entrada] + [e[:, :, None] for e in extras], axis=2)
    return entrada

def ler_tudo(r, pasta_cache):
    rotulo, antes, depois, invalidos = ler_recorte(r, pasta_cache)
    rad = par_radar(r, pasta_cache)
    if rad is None:
        return None
    invalidos = invalidos | np.isnan(rad["vh_antes"]) | np.isnan(rad["vh_depois"])
    return rotulo, antes, depois, invalidos, rad

def pre_treinar(recortes, com_radar, seed, pasta_cache):
    rng = np.random.RandomState(seed)
    amostras = []
    for r in recortes:
        dados = ler_tudo(r, pasta_cache)
        if dados is None:
            continue
        _, antes, depois, invalidos, rad = dados
        entrada = montar(antes, depois, rad, com_radar)
        validos = np.flatnonzero(~invalidos.ravel())
        sel = rng.choice(validos, int(round(len(validos) * 0.02)), replace=False)
        amostras.append(enchenteSOM.extrair_janelas(entrada, 1, sel))
    treino = np.concatenate(amostras)
    rng.shuffle(treino)
    enchenteSOM.somSeg.rand.seed(seed)
    SOM = enchenteSOM.somSeg.create_map(enchenteSOM.SOM_SIZE, 1, treino.shape[1])
    return enchenteSOM.somSeg.train_SOM(SOM, treino)

def avaliar(args):
    r, modelos, pasta_cache = args
    arquivo = os.path.join(pasta_cache, "radar_par_resultados", r["recorte"] + ".json")
    if os.path.exists(arquivo):
        with open(arquivo) as f:
            return json.load(f)
    dados = ler_tudo(r, pasta_cache)
    if dados is None:
        res = {"recorte": r["recorte"], "status": "sem par de radar"}
    else:
        rotulo, antes, depois, invalidos, rad = dados
        validos = (rotulo >= 0) & ~invalidos
        ref = (rotulo == 1) & ~ler_jrc(r["recorte"], pasta_cache)
        res = {"recorte": r["recorte"], "pais": r["pais"], "status": "ok",
               "pct_alagado_ref": round(100 * float(ref[validos].mean()), 2),
               "data_antes_radar": str(rad["data_antes_radar"]), "data_depois_radar": str(rad["data_depois_radar"]),
               "variacoes": {}}
        for nome, mapa in mapas_pixel(antes, depois, rad).items():
            res["variacoes"][nome] = [contar(mapa == 2, ref, validos)]
        for com_radar_entrada, lista in modelos.items():
            entrada = montar(antes, depois, rad, com_radar_entrada)
            for SOM in lista:
                result = enchenteSOM.segmentar(SOM, entrada, 1)
                for regra_radar in (False, True):
                    nome = ("SOM com radar na entrada" if com_radar_entrada else "SOM óptico") + \
                           (" + mudança no radar" if regra_radar else "")
                    classe = classificar_unidades(result, invalidos, antes, depois, rad, enchenteSOM.SOM_SIZE, regra_radar)
                    res["variacoes"].setdefault(nome, []).append(contar(classe[result] == 2, ref, validos))
    with open(arquivo, "w") as f:
        json.dump(res, f)
    print("%-22s %s" % (r["recorte"], res["status"] if res["status"] != "ok" else
          "alagado %5.1f%% radar %s -> %s" % (res["pct_alagado_ref"], res["data_antes_radar"], res["data_depois_radar"])),
          flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o par de radar Sentinel-1 antes/depois na detecção do alagamento")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--sementes", type=int, default=3)
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    for sub in ("radar_par", "radar_par_resultados"):
        os.makedirs(os.path.join(args.pasta_cache, sub), exist_ok=True)

    por_evento = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.pasta_cache, "resultados", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    escolhidos = {p: escolher_pre_treino(l) for p, l in por_evento.items()}
    excluidos = set(r["recorte"] for r in escolhidos.values()) | set(PRE_TREINO)
    imagens = [e for e in escolhidos.values() if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX]
    modelos = {c: [pre_treinar(imagens, c, s, args.pasta_cache) for s in range(args.sementes)] for c in (False, True)}
    print("Pré-treinos prontos (%d imagens)" % len(imagens), flush=True)

    tarefas = [(r, modelos, args.pasta_cache) for p in sorted(por_evento) for r in por_evento[p]
               if r["recorte"] not in excluidos]
    with Pool(args.processos) as pool:
        todos = pool.map(avaliar, tarefas, chunksize=1)
    resultados = [r for r in todos if r["status"] == "ok"]

    nomes = list(resultados[0]["variacoes"])
    def tabela_de(sel):
        t = {"n": len(sel)}
        for n in nomes:
            k = len(sel[0]["variacoes"][n]) if sel else 0
            ms = [metricas(somar([r["variacoes"][n][s] for r in sel])) for s in range(k)]
            t[n] = {m: float(np.mean([x[m] for x in ms])) for m in ms[0]} if ms else None
        return t
    tabela = tabela_de(resultados)
    eventos = sorted(set(r["pais"] for r in resultados))
    tabela_eventos = {p: tabela_de([r for r in resultados if r["pais"] == p]) for p in eventos}
    com_alag = [r for r in resultados if r["pct_alagado_ref"] >= 1]
    falhas = {n: sum(1 for r in com_alag if np.mean([metricas(c)["IoU"] for c in r["variacoes"][n]]) < 0.05) for n in nomes}
    dias = [(date.fromisoformat(r["data_depois_radar"]) - date.fromisoformat(r["data_antes_radar"])).days for r in resultados]

    print("\n%d recortes com par de radar (de %d); intervalo entre as cenas de radar: mediana %d dias"
          % (len(resultados), len(todos), np.median(dias)))
    print("%-42s %6s %6s %6s %6s %7s %9s" % ("Variação", "IoU", "F1", "prec.", "sens.", "falhas", "Paraguai"))
    for n in nomes:
        t = tabela[n]
        print("%-42s %6.3f %6.3f %6.3f %6.3f %7d %9.3f" % (n, t["IoU"], t["F1"], t["precisao"], t["sensibilidade"],
              falhas[n], tabela_eventos["Paraguay"][n]["IoU"]))
    print("\nIoU por evento:")
    print("%-10s %4s " % ("Evento", "n") + " ".join("%11s" % n[:11] for n in nomes))
    for p in eventos:
        print("%-10s %4d " % (p, tabela_eventos[p]["n"]) + " ".join("%11.3f" % tabela_eventos[p][n]["IoU"] for n in nomes))

    with open(os.path.join(args.saida, "radar_bitemporal.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes": len(resultados), "recortes_sem_par": len(todos) - len(resultados),
                   "limiar_vh_db": LIMIAR_VH, "delta_vh_db": DELTA_VH, "sementes": args.sementes,
                   "intervalo_radar_dias_mediana": float(np.median(dias)), "alagamento": tabela,
                   "alagamento_por_evento": tabela_eventos, "falhas_recortes_com_alagamento": falhas,
                   "recortes_com_alagamento": len(com_alag)}, f, ensure_ascii=False, indent=1)

    cores = ["#ff9f1c", "#7f7f7f", "#bdbdbd", "#b35806", "#bcbddc", "#6a51a3", "#9ecae1", "#08519c"]
    fig, ax = plt.subplots(1, 2, figsize=(25, 6.5), gridspec_kw=dict(width_ratios=[2.4, 1]))
    grupos = ["Todos"] + eventos
    w = 0.85 / len(nomes)
    for k, n in enumerate(nomes):
        v = [tabela[n]["IoU"]] + [tabela_eventos[p][n]["IoU"] for p in eventos]
        ax[0].bar(np.arange(len(grupos)) + (k - (len(nomes) - 1) / 2) * w, v, w, color=cores[k % len(cores)], label=n,
                  edgecolor="k", lw=0.4)
    ax[0].set_xticks(np.arange(len(grupos)))
    ax[0].set_xticklabels(["Todos\n(%d)" % len(resultados)] + ["%s\n(%d)" % (p, tabela_eventos[p]["n"]) for p in eventos])
    ax[0].set_ylabel("IoU do alagamento")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8, ncol=4, loc="upper left")
    ax[0].set_title("Por evento (número de recortes)")
    for k, n in enumerate(nomes):
        ax[1].scatter(tabela[n]["sensibilidade"], tabela[n]["precisao"], s=150, color=cores[k % len(cores)],
                      edgecolor="k", label=n)
    ax[1].set_xlabel("Sensibilidade")
    ax[1].set_ylabel("Precisão")
    ax[1].set_xlim(0.2, 1)
    ax[1].set_ylim(0.2, 1)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Precisão x sensibilidade")
    fig.suptitle("Sen1Floods11: par de radar Sentinel-1 antes/depois (RTC, mesma órbita) na detecção do alagamento - "
                 "VH < %g dB e queda > %g dB" % (LIMIAR_VH, -DELTA_VH), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "radar_bitemporal.png"), dpi=85)
    print("Resultados salvos em", args.saida)
