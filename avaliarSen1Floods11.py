# -*- coding: utf-8 -*-
"""
Avalia o SOM bi-temporal, o NDWI e o MNDWI contra os rótulos de água feitos à
mão do Sen1Floods11 (446 recortes de 512x512 pixels, ~10 m, em 11 eventos de
inundação).

Para cada recorte:
1. Baixa o rótulo (LabelHand: -1 sem dado, 0 não água, 1 água) e, para
   comparação, o rótulo automático por limiar de Otsu no radar Sentinel-1
   (S1OtsuLabelHand), do bucket público do Sen1Floods11.
2. Busca no Planetary Computer a cena Sentinel-2 L2A da data do evento (a de
   menos pixels inválidos no recorte, até 1 dia de diferença) e a cena de antes
   mais próxima, do mesmo tile, com no máximo 10% de pixels inválidos.
3. Lê as bandas das duas cenas reprojetadas para a grade do rótulo e aplica os
   três métodos. Água = água permanente ou alagamento na data do evento (o
   rótulo do Sen1Floods11 não separa as duas).
4. Conta acertos e erros nos pixels válidos (rótulo diferente de -1 e sem
   nuvem/sombra nas cenas Sentinel-2).

O resultado de cada recorte fica em <pasta_cache>/resultados/<recorte>.json, e
uma nova execução reaproveita os já feitos. No final: resultados/sen1floods11/
com as métricas por método e por evento (IoU, F1, precisão, sensibilidade) e
figuras com exemplos.

Uso:
    python avaliarSen1Floods11.py                   # todos os recortes
    python avaliarSen1Floods11.py --paises Paraguay Mekong
"""

import argparse
import glob
import json
import os
import urllib.request
from datetime import datetime, timedelta
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

import enchenteSOM
import sentinelUtils


URL_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
METODOS = ["SOM", "NDWI", "MNDWI", "S1 Otsu"]

# Nome do país no metadado do Sen1Floods11 para o prefixo dos recortes
PAIS_METADADO = {"Mekong": "Cambodia"}


def baixar(url, arquivo):
    if not os.path.exists(arquivo):
        os.makedirs(os.path.dirname(arquivo), exist_ok=True)
        urllib.request.urlretrieve(url, arquivo + ".tmp")
        os.replace(arquivo + ".tmp", arquivo)
    return arquivo

def listar_recortes(pasta_cache):
    # Nomes dos recortes a partir do catálogo STAC do Sen1Floods11
    arquivo = baixar(URL_BASE + "/catalog/sen1floods11_hand_labeled_label/collection.json",
                     os.path.join(pasta_cache, "collection.json"))
    with open(arquivo) as f:
        colecao = json.load(f)
    return sorted(set(l["href"].split("/")[-2].replace("_label", "") for l in colecao["links"] if l["rel"] == "item"))

def datas_s2(pasta_cache):
    arquivo = baixar(URL_BASE + "/Sen1Floods11_Metadata.geojson", os.path.join(pasta_cache, "metadata.geojson"))
    with open(arquivo) as f:
        return {ft["properties"]["location"]: ft["properties"]["s2_date"].replace("/", "-")
                for ft in json.load(f)["features"]}

def ler_rotulo(arquivo):
    with rasterio.open(arquivo) as src:
        return src.read(1), src.crs, src.transform, src.width, src.height, list(src.bounds)

# Escolhe a cena da data do evento com menos pixels inválidos no recorte (até 1 dia de diferença)
def cena_evento(bbox, data):
    dia = datetime.strptime(data, "%Y-%m-%d")
    candidatas = sentinelUtils.buscar_cenas(bbox, dia - timedelta(days=1), dia + timedelta(days=2))
    if not candidatas:
        return None, None
    notas = [(sentinelUtils.fracao_invalida(it, bbox, resolucao=20) + 0.05 * abs((it.datetime.replace(tzinfo=None) - dia).days),
              it) for it in candidatas]
    nota, item = min(notas, key=lambda n: n[0])
    return item, nota

def contar(previsto, referencia, validos):
    p, r = previsto[validos], referencia[validos]
    return {"vp": int(np.sum(p & r)), "fp": int(np.sum(p & ~r)), "fn": int(np.sum(~p & r)), "vn": int(np.sum(~p & ~r))}

def avaliar_recorte(args):
    recorte, data, pasta_cache = args
    arquivo_resultado = os.path.join(pasta_cache, "resultados", recorte + ".json")
    if os.path.exists(arquivo_resultado):
        with open(arquivo_resultado) as f:
            anterior = json.load(f)
        # Erros de rede ou de código são refeitos; os recortes sem dados suficientes, não
        if anterior["status"] == "ok" or anterior["status"].startswith(("erro: Menos de", "erro: Nenhuma cena")):
            return anterior
    res = {"recorte": recorte, "pais": recorte.split("_")[0], "data_evento": data}
    try:
        rotulo, crs, transform, largura, altura, bounds = ler_rotulo(baixar(
            "%s/data/flood_events/HandLabeled/LabelHand/%s_LabelHand.tif" % (URL_BASE, recorte),
            os.path.join(pasta_cache, "rotulos", recorte + "_LabelHand.tif")))
        otsu = ler_rotulo(baixar(
            "%s/data/flood_events/HandLabeled/S1OtsuLabelHand/%s_S1OtsuLabelHand.tif" % (URL_BASE, recorte),
            os.path.join(pasta_cache, "rotulos", recorte + "_S1OtsuLabelHand.tif")))[0]
        bbox = bounds  # Já em graus (EPSG:4326)

        item_depois, invalida_depois = cena_evento(bbox, data)
        if item_depois is None:
            raise RuntimeError("Nenhuma cena Sentinel-2 na data do evento")
        item_antes, invalida_antes = sentinelUtils.selecionar_cena_antes(bbox, data, item_depois, dias_max=180)
        res.update(cena_antes=item_antes.id, cena_depois=item_depois.id,
                   data_antes=item_antes.datetime.strftime("%Y-%m-%d"), data_depois=item_depois.datetime.strftime("%Y-%m-%d"))

        grade = (crs, transform, largura, altura)
        refl_antes = sentinelUtils.ler_bandas_na_grade(item_antes, enchenteSOM.BANDAS, *grade)
        refl_depois = sentinelUtils.ler_bandas_na_grade(item_depois, enchenteSOM.BANDAS, *grade)
        invalidos = (sentinelUtils.ler_mascara_nuvens_na_grade(item_antes, *grade)
                     | sentinelUtils.ler_mascara_nuvens_na_grade(item_depois, *grade))
        validos = (rotulo >= 0) & ~invalidos
        res["pct_validos"] = round(100 * float(validos.mean()), 1)
        if validos.mean() < 0.2:
            raise RuntimeError("Menos de 20%% de pixels válidos (%.0f%%)" % (100 * validos.mean()))

        # SOM bi-temporal e limiares diretos; água = água permanente ou alagamento
        entrada, _ = enchenteSOM.montar_entrada(refl_antes, refl_depois)
        _, result = enchenteSOM.treinar_e_segmentar(entrada, invalidos)
        classe_unidade, _ = enchenteSOM.classificar_unidades(result, invalidos, refl_antes, refl_depois, enchenteSOM.SOM_SIZE)
        mapas = {"SOM": classe_unidade[result],
                 "NDWI": enchenteSOM.mapa_indice(refl_antes, refl_depois, 0, "NDWI"),
                 "MNDWI": enchenteSOM.mapa_indice(refl_antes, refl_depois, 0, "MNDWI")}
        agua = {m: np.isin(mp, [1, 2]) for m, mp in mapas.items()}
        agua["S1 Otsu"] = otsu == 1
        validos_otsu = validos & (otsu >= 0)
        referencia = rotulo == 1
        res["contagens"] = {m: contar(a, referencia, validos_otsu if m == "S1 Otsu" else validos) for m, a in agua.items()}
        res["pct_agua_rotulo"] = round(100 * float(referencia[validos].mean()), 1)

        # Mapas pequenos para as figuras de exemplo
        np.savez_compressed(os.path.join(pasta_cache, "mapas", recorte + ".npz"), rotulo=rotulo, invalidos=invalidos,
                            rgb=(enchenteSOM.cor_verdadeira(refl_depois, 0.2) * 255).astype(np.uint8),
                            rgb_antes=(enchenteSOM.cor_verdadeira(refl_antes, 0.2) * 255).astype(np.uint8),
                            **{m.replace(" ", "_"): a for m, a in agua.items()})
        res["status"] = "ok"
    except Exception as erro:
        res["status"] = "erro: %s" % erro
    with open(arquivo_resultado, "w") as f:
        json.dump(res, f, indent=1)
    print("%-22s %s" % (recorte, res["status"] if res["status"] != "ok" else
          " ".join("%s IoU %.2f" % (m, iou(res["contagens"][m])) for m in METODOS)), flush=True)
    return res

def iou(c):
    return c["vp"] / max(c["vp"] + c["fp"] + c["fn"], 1)

def metricas(c):
    vp, fp, fn, vn = c["vp"], c["fp"], c["fn"], c["vn"]
    return {"IoU": vp / max(vp + fp + fn, 1), "F1": 2 * vp / max(2 * vp + fp + fn, 1),
            "precisao": vp / max(vp + fp, 1), "sensibilidade": vp / max(vp + fn, 1),
            "acuracia": (vp + vn) / max(vp + fp + fn + vn, 1)}

def somar(lista):
    return {k: sum(c[k] for c in lista) for k in ("vp", "fp", "fn", "vn")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia SOM, NDWI e MNDWI contra os rótulos do Sen1Floods11")
    parser.add_argument("--pasta-cache", default="cache_sen1floods11")
    parser.add_argument("--saida", default=os.path.join("resultados", "sen1floods11"))
    parser.add_argument("--paises", nargs="*", help="Prefixos dos recortes (ex.: Paraguay Mekong); padrão: todos")
    parser.add_argument("--processos", type=int, default=4)
    args = parser.parse_args()
    for sub in ("resultados", "mapas", "rotulos"):
        os.makedirs(os.path.join(args.pasta_cache, sub), exist_ok=True)
    os.makedirs(args.saida, exist_ok=True)

    datas = datas_s2(args.pasta_cache)
    recortes = [r for r in listar_recortes(args.pasta_cache) if not args.paises or r.split("_")[0] in args.paises]
    tarefas = [(r, datas[PAIS_METADADO.get(r.split("_")[0], r.split("_")[0])], args.pasta_cache) for r in recortes]
    print("%d recortes" % len(tarefas), flush=True)
    with Pool(args.processos) as pool:
        resultados = pool.map(avaliar_recorte, tarefas, chunksize=1)

    ok = [r for r in resultados if r["status"] == "ok"]
    paises = sorted(set(r["pais"] for r in ok))
    tabela = {"Todos": {m: metricas(somar([r["contagens"][m] for r in ok])) for m in METODOS}}
    for p in paises:
        tabela[p] = {m: metricas(somar([r["contagens"][m] for r in ok if r["pais"] == p])) for m in METODOS}
    n_por_pais = {p: sum(1 for r in ok if r["pais"] == p) for p in paises}
    # Proporção de recortes em que cada método tem a maior IoU (entre SOM, NDWI e MNDWI)
    com_agua = [r for r in ok if r["contagens"]["SOM"]["vp"] + r["contagens"]["SOM"]["fn"] > 0]
    vitorias = {m: 0 for m in METODOS[:3]}
    for r in com_agua:
        vitorias[max(METODOS[:3], key=lambda m: iou(r["contagens"][m]))] += 1

    # Métricas por fração de água no rótulo: o SOM perde a água quando ela é pouca no recorte
    faixas = [(0, 1), (1, 5), (5, 15), (15, 40), (40, 101)]
    por_faixa = {}
    for a, b in faixas:
        sel = [r for r in ok if a <= r["pct_agua_rotulo"] < b]
        por_faixa["%d-%d%%" % (a, min(b, 100))] = {"n": len(sel), **{m: metricas(somar([r["contagens"][m] for r in sel]))
                                                                     if sel else None for m in METODOS}}

    print("\n%d de %d recortes avaliados" % (len(ok), len(resultados)))
    print("%-12s %5s " % ("Evento", "n") + " ".join("%12s" % m for m in METODOS) + "   (IoU da água)")
    for p in ["Todos"] + paises:
        print("%-12s %5s " % (p, len(ok) if p == "Todos" else n_por_pais[p])
              + " ".join("%12.3f" % tabela[p][m]["IoU"] for m in METODOS))
    print("\nIoU da água por fração de água no rótulo:")
    for faixa, v in por_faixa.items():
        print("  %-8s n=%3d " % (faixa, v["n"]) + " ".join("%s %.3f" % (m, v[m]["IoU"]) for m in METODOS if v[m]))
    print("\nMaior IoU por recorte (%d recortes com água):" % len(com_agua),
          ", ".join("%s %d" % (m, v) for m, v in vitorias.items()))

    with open(os.path.join(args.saida, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump({"recortes_avaliados": len(ok), "recortes_total": len(resultados), "recortes_por_evento": n_por_pais,
                   "metricas": tabela, "metricas_por_fracao_de_agua": por_faixa, "maior_iou_por_recorte": vitorias,
                   "erros": {r["recorte"]: r["status"] for r in resultados if r["status"] != "ok"}},
                  f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.saida, "por_recorte.csv"), "w", encoding="utf-8") as f:
        f.write("recorte,pais,data_antes,data_depois,pct_validos,pct_agua_rotulo," + ",".join("iou_" + m.replace(" ", "_") for m in METODOS) + "\n")
        for r in ok:
            f.write("%s,%s,%s,%s,%s,%s,%s\n" % (r["recorte"], r["pais"], r["data_antes"], r["data_depois"], r["pct_validos"],
                    r["pct_agua_rotulo"], ",".join("%.4f" % iou(r["contagens"][m]) for m in METODOS)))

    # Figura 1: IoU por evento e método
    cores = {"SOM": "#3fa9f5", "NDWI": "#e7298a", "MNDWI": "#ff9f1c", "S1 Otsu": "#7f7f7f"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(22, 5.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    grupos = ["Todos"] + paises
    w = 0.2
    for k, m in enumerate(METODOS):
        v = [tabela[g][m]["IoU"] for g in grupos]
        ax.bar(np.arange(len(grupos)) + (k - 1.5) * w, v, w, color=cores[m], label=m, edgecolor="k", lw=0.4)
    ax.set_xticks(np.arange(len(grupos)))
    ax.set_xticklabels(["%s\n(%d)" % (g, len(ok) if g == "Todos" else n_por_pais[g]) for g in grupos], fontsize=10)
    ax.set_ylabel("IoU da água (todos os pixels do evento)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4)
    ax.set_title("Por evento (número de recortes entre parênteses)")
    nomes_faixa = list(por_faixa)
    for k, m in enumerate(METODOS):
        v = [por_faixa[f][m]["IoU"] if por_faixa[f][m] else 0 for f in nomes_faixa]
        ax2.bar(np.arange(len(nomes_faixa)) + (k - 1.5) * w, v, w, color=cores[m], label=m, edgecolor="k", lw=0.4)
    ax2.set_xticks(np.arange(len(nomes_faixa)))
    ax2.set_xticklabels(["%s\n(%d)" % (f, por_faixa[f]["n"]) for f in nomes_faixa], fontsize=10)
    ax2.set_xlabel("Fração de água no rótulo do recorte")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Por fração de água no recorte")
    fig.suptitle("Sen1Floods11 (rótulos à mão): SOM bi-temporal x NDWI x MNDWI (Sentinel-2) e Otsu no radar Sentinel-1",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.saida, "iou_por_evento.png"), dpi=90)

    # Figura 2: exemplos onde o SOM mais supera o NDWI e onde mais perde
    dif = sorted(com_agua, key=lambda r: iou(r["contagens"]["SOM"]) - iou(r["contagens"]["NDWI"]))
    exemplos = dif[-4:][::-1] + dif[:2]
    if exemplos:
        fig, ax = plt.subplots(len(exemplos), 6, figsize=(21, 3.7 * len(exemplos)), subplot_kw=dict(xticks=[], yticks=[]))
        ax = np.atleast_2d(ax)
        for k, r in enumerate(exemplos):
            d = np.load(os.path.join(args.pasta_cache, "mapas", r["recorte"] + ".npz"))
            ax[k, 0].imshow(d["rgb_antes"])
            ax[k, 0].set_title("Antes %s" % r["data_antes"], fontsize=10)
            ax[k, 1].imshow(d["rgb"])
            ax[k, 1].set_title("Evento %s" % r["data_depois"], fontsize=10)
            ax[k, 2].imshow(np.where(d["rotulo"] < 0, 0.5, d["rotulo"]), cmap=ListedColormap(["#e8e8e8", "#bdbdbd", "#08306b"]),
                            vmin=0, vmax=1)
            ax[k, 2].set_title("Rótulo à mão", fontsize=10)
            validos = (d["rotulo"] >= 0) & ~d["invalidos"]
            for m, nome in enumerate(["SOM", "NDWI", "MNDWI"]):
                a = d[nome]
                ref = d["rotulo"] == 1
                erro = np.zeros(a.shape, dtype=int)  # 0 não água certo, 1 água certa, 2 falso positivo, 3 falso negativo
                erro[a & ref] = 1
                erro[a & ~ref] = 2
                erro[~a & ref] = 3
                erro[~validos] = 4
                ax[k, m + 3].imshow(erro, cmap=ListedColormap(["#f0f0f0", "#3fa9f5", "#d95f0e", "#ffd92f", "#ffffff"]),
                                    vmin=-0.5, vmax=4.5, interpolation="nearest")
                ax[k, m + 3].set_title("%s - IoU %.2f" % (nome, iou(r["contagens"][nome])), fontsize=10)
            ax[k, 0].set_ylabel(r["recorte"], fontsize=10)
        fig.suptitle("Exemplos: maiores vantagens (4 primeiras linhas) e desvantagens (2 últimas) do SOM sobre o NDWI\n"
                     "Azul: água certa; laranja: falso positivo; amarelo: falso negativo; branco: sem dado ou nuvem",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (3.7 * len(exemplos))))
        fig.savefig(os.path.join(args.saida, "exemplos.png"), dpi=70)
    print("Resultados salvos em", args.saida)
