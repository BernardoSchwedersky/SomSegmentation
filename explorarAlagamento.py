# -*- coding: utf-8 -*-
"""
Explora as áreas vizinhas com alagamento em torno de um evento, aumentando a
grade a partir da grade 3x3 inicial.

Cada área processada (SOM bi-temporal e baseline NDWI, como em
processarEventos.py) que tiver pelo menos limiar_expansao_km2 de alagamento
pelo SOM coloca as suas 8 vizinhas na fila. As áreas são processadas em ondas,
em paralelo, até que as bordas não tenham mais alagamento ou até raio_max áreas
de distância do centro. Áreas com nuvens ou sem dado demais são puladas e não
expandem a grade.

O evento é escolhido com --evento (localidade de eventos_inundacao.csv): o
centro vem da tabela e as datas das cenas de antes e de depois vêm do resultado
do evento em resultados/<evento>.json (gerado por processarEventos.py). Sem
--evento, usa Pelotas com as datas abaixo.

Com --incluir LAT_MIN LAT_MAX LON_MIN LON_MAX, todas as áreas de um retângulo
entram na fila desde o início (ex.: para unir a grade a uma cidade vizinha
separada por água permanente ou por áreas com nuvens).

As cenas de antes e de depois são de datas fixas. Como uma região grande pode
cruzar a borda entre tiles do Sentinel-2, cada área usa o tile (da mesma
aquisição) com menos pixels inválidos.

Saída: <pasta_saida>/<nome>.png (figura de comparação do mosaico) e
<pasta_saida>/<nome>.json (resultado de cada área e totais).

As funções estão em comparacaoGrade.py, enchenteSOM.py e sentinelUtils.py.
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timedelta
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import comparacaoGrade
import enchenteSOM
import processarEventos
import sentinelUtils


# Evento padrão: Pelotas - RS, cenas de antes e de depois
nome = "pelotas_expandido"
titulo_evento = "Pelotas, Brasil - evento em 26/05/2024"
centro_lon, centro_lat = -52.3265, -31.7725
data_antes = "2024-02-01"
data_depois = "2024-05-26"

# Parâmetros da expansão
meia_largura_graus = 0.025    # Cada área tem ~5,5 km de lado
raio_max = 12                 # No máximo 12 áreas do centro em cada direção (grade de até 25x25)
limiar_expansao_km2 = 0.5     # Alagamento (SOM) mínimo de uma área para expandir para as vizinhas
nuvens_max_area = 20          # % máxima de nuvens ou sem dado para processar uma área

# Parâmetros de ajuste do SOM e do baseline
patch_size = 1
som_size = 12
epochs = 3
dataset_percentage = 0.02
limiar_agua = 0


# Retângulo da área (di, dj): di linhas ao norte e dj colunas a leste do centro
def bbox_area(di, dj):
    return sentinelUtils.criar_bbox(centro_lon + dj * 2 * meia_largura_graus,
                                    centro_lat + di * 2 * meia_largura_graus, meia_largura_graus)

def nome_area(di, dj):
    return "%s%d %s%d" % ("N" if di >= 0 else "S", abs(di), "L" if dj >= 0 else "O", abs(dj))

# Pares (antes, depois) das duas datas, um por tile que tem as duas cenas, com o
# retângulo coberto pelo tile (para testar em cada área só os tiles que a cobrem)
def pares_por_tile(bbox, data_depois):
    def por_tile(data):
        dia = datetime.strptime(data, "%Y-%m-%d")
        return {item.properties["s2:mgrs_tile"]: item
                for item in sentinelUtils.buscar_cenas(bbox, dia, dia + timedelta(days=1))
                if item.datetime.date() == dia.date()}
    antes, depois = por_tile(data_antes), por_tile(data_depois)
    return [(antes[t].id, depois[t].id, list(depois[t].bbox)) for t in sorted(antes) if t in depois]

def intersecta(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

# Processa uma área: escolhe o tile com menos pixels inválidos nas duas datas
# e aplica os dois métodos. grupos tem os pares da data de depois principal e,
# em seguida, os das datas alternativas, usadas só se a área tiver nuvens ou
# falta de dados demais nas anteriores. As cenas são buscadas de novo aqui, para
# que as URLs assinadas não expirem durante processamentos longos. O resultado
# de cada área fica salvo na pasta de cache, e uma nova execução o reaproveita
def processar(args):
    di, dj, grupos, pasta_cache = args
    datas = [g[0] for g in grupos]
    arquivo_resultado = os.path.join(pasta_cache, "resultado_%d_%d.pkl" % (di, dj))
    if os.path.exists(arquivo_resultado):
        with open(arquivo_resultado, "rb") as f:
            area = pickle.load(f)
        if "mapa_som" in area or set(datas) <= set(area.get("datas_testadas", datas[:1])):
            return area
    bbox = bbox_area(di, dj)
    melhor = None
    for data, pares in grupos:
        for id_antes, id_depois, bbox_tile in pares:
            if not intersecta(bbox, bbox_tile):
                continue
            item_antes, item_depois = sentinelUtils.buscar_item(id_antes), sentinelUtils.buscar_item(id_depois)
            invalida = max(sentinelUtils.fracao_invalida(item_antes, bbox),
                           sentinelUtils.fracao_invalida(item_depois, bbox))
            if melhor is None or invalida < melhor[0]:
                melhor = (invalida, item_antes, item_depois)
        if melhor is not None and melhor[0] <= nuvens_max_area / 100:
            break  # A data principal (ou a alternativa anterior) já serve
    if melhor is None:
        area = {"nome": nome_area(di, dj), "bbox": bbox, "pct_nuvens": 100.0, "tile": None}
    else:
        _, item_antes, item_depois = melhor
        area = comparacaoGrade.processar_area(
            item_antes, item_depois, bbox, nome_area(di, dj), os.path.join(pasta_cache, "area_%d_%d.npz" % (di, dj)),
            nuvens_max_area, patch_size, som_size, epochs, dataset_percentage, limiar_agua)
        area["tile"] = item_depois.properties["s2:mgrs_tile"]
        area["data_depois"] = item_depois.datetime.strftime("%Y-%m-%d")
    area.update(di=di, dj=dj, datas_testadas=datas)
    with open(arquivo_resultado, "wb") as f:
        pickle.dump(area, f)
    return area

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explora as áreas vizinhas com alagamento")
    parser.add_argument("--pasta-saida", default=os.path.join("resultados", "expansao"))
    parser.add_argument("--pasta-cache", default="cache_explorar")
    parser.add_argument("--processos", type=int, default=4)
    parser.add_argument("--raio-max", type=int, default=raio_max, help="Distância máxima, em áreas, a partir do centro")
    parser.add_argument("--evento", help="Localidade de eventos_inundacao.csv (padrão: Pelotas)")
    parser.add_argument("--data-antes", help="Data da cena de antes (padrão: a do resultado do evento)")
    parser.add_argument("--data-depois", help="Data da cena de depois (padrão: a do resultado do evento)")
    parser.add_argument("--rgb-max", type=float, default=enchenteSOM.RGB_MAX,
                        help="Reflectância que vira branco na figura (use ~0,35 em deserto)")
    parser.add_argument("--incluir", nargs=4, type=float, action="append", default=[],
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Retângulo cujas áreas entram na fila desde o início (pode repetir), ex.: para "
                             "unir a grade a uma cidade vizinha separada por água permanente ou nuvens")
    parser.add_argument("--datas-depois-alternativas", nargs="*", default=[], metavar="AAAA-MM-DD",
                        help="Datas de depois usadas, em ordem, nas áreas com nuvens demais na data principal")
    args = parser.parse_args()
    raio_max = args.raio_max
    if args.evento:
        evento = next(e for e in processarEventos.ler_eventos("eventos_inundacao.csv") if e["localidade"] == args.evento)
        arquivo_evento = os.path.join("resultados", processarEventos.nome_arquivo(args.evento) + ".json")
        resultado_evento = {}
        if os.path.exists(arquivo_evento):
            with open(arquivo_evento, encoding="utf-8") as f:
                resultado_evento = json.load(f)
        nome = processarEventos.nome_arquivo(args.evento) + "_expandido"
        titulo_evento = "%s, %s - evento em %s" % (evento["localidade"], evento["pais"],
                                                   "/".join(reversed(evento["data"].split("-"))))
        centro_lon, centro_lat = float(evento["longitude"]), float(evento["latitude"])
        data_antes = args.data_antes or resultado_evento["data_antes"]
        data_depois = args.data_depois or resultado_evento["data_depois"]
    elif args.data_antes or args.data_depois:
        data_antes, data_depois = args.data_antes or data_antes, args.data_depois or data_depois
    enchenteSOM.RGB_MAX = args.rgb_max  # Herdado pelos processos do Pool
    print("%s: antes %s, depois %s" % (titulo_evento, data_antes, data_depois), flush=True)
    os.makedirs(args.pasta_saida, exist_ok=True)
    pasta_cache = os.path.join(args.pasta_cache, nome)
    os.makedirs(pasta_cache, exist_ok=True)

    regiao = sentinelUtils.criar_bbox(centro_lon, centro_lat, (2 * raio_max + 1) * meia_largura_graus)
    grupos = [(d, pares_por_tile(regiao, d)) for d in [data_depois] + args.datas_depois_alternativas]
    pares = grupos[0][1]
    for d, p in grupos:
        print("Depois em %s - tiles com as duas cenas:" % d, [x[0].split("_")[4] for x in p], flush=True)

    areas = {}
    fila = {(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)}
    passo = 2 * meia_largura_graus
    for lat_min, lat_max, lon_min, lon_max in args.incluir:
        for di in range(int(np.ceil((lat_min - centro_lat) / passo)), int(np.floor((lat_max - centro_lat) / passo)) + 1):
            for dj in range(int(np.ceil((lon_min - centro_lon) / passo)), int(np.floor((lon_max - centro_lon) / passo)) + 1):
                fila.add((di, dj))
    fila = sorted(fila)
    onda = 0
    with Pool(args.processos) as pool:
        while fila:
            print("\n#### Onda %d: %d áreas" % (onda, len(fila)), flush=True)
            for area in pool.map(processar, [(di, dj, grupos, pasta_cache) for di, dj in fila]):
                areas[(area["di"], area["dj"])] = area
            nova = set()
            for (di, dj), area in areas.items():
                if "mapa_som" not in area or area["areas_som"][2] < limiar_expansao_km2:
                    continue
                for vi in (-1, 0, 1):
                    for vj in (-1, 0, 1):
                        v = (di + vi, dj + vj)
                        if v not in areas and max(abs(v[0]), abs(v[1])) <= raio_max:
                            nova.add(v)
            fila = sorted(nova)
            onda += 1

    lista = list(areas.values())
    t = comparacaoGrade.totais(lista)
    print("\nÁreas exploradas: %d (%d processadas) | alagamento SOM %.1f km², NDWI %.1f km²"
          % (len(lista), t["areas_processadas"], t["som_alagamento_km2"], t["ndwi_alagamento_km2"]))

    # Para a figura, só as cenas de referência (as datas são as mesmas em todos os tiles)
    item_antes, item_depois = sentinelUtils.buscar_item(pares[0][0]), sentinelUtils.buscar_item(pares[0][1])
    titulo = ("%s: áreas vizinhas com alagamento (%d áreas) - SOM bi-temporal x baseline NDWI (NDWI > %g)"
              % (titulo_evento, len(lista), limiar_agua))
    # Áreas que usaram uma data de depois alternativa (por causa de nuvens na principal)
    alternativas = {}
    for a in lista:
        if a.get("data_depois", data_depois) != data_depois:
            alternativas[a["data_depois"]] = alternativas.get(a["data_depois"], 0) + 1
    rotulo_depois = "Depois: " + "/".join(reversed(data_depois.split("-")))
    if alternativas:
        rotulo_depois += " (%s)" % ", ".join("%s em %d áreas" % ("/".join(reversed(d.split("-"))), n)
                                             for d, n in sorted(alternativas.items()))
    fig = comparacaoGrade.figura_comparacao(lista, item_antes, item_depois, centro_lat, limiar_agua, titulo,
                                            os.path.join(args.pasta_saida, nome + ".png"), dpi=110,
                                            mostrar_nomes=False, rotulo_depois=rotulo_depois)
    plt.close(fig)

    resultado = {"nome": nome, "data_antes": data_antes, "data_depois": data_depois,
                 "datas_depois_alternativas": args.datas_depois_alternativas, "raio_max": raio_max,
                 "limiar_expansao_km2": limiar_expansao_km2, "totais": t,
                 "areas": [{"area": a["nome"], "di": a["di"], "dj": a["dj"], "tile": a["tile"], "bbox": a["bbox"],
                            "data_depois": a.get("data_depois"),
                            "pct_nuvens": round(a["pct_nuvens"], 1),
                            **({"som_permanente_km2": round(a["areas_som"][1], 3),
                                "som_alagamento_km2": round(a["areas_som"][2], 3),
                                "ndwi_permanente_km2": round(a["areas_ndwi"][1], 3),
                                "ndwi_alagamento_km2": round(a["areas_ndwi"][2], 3),
                                "iou_agua": round(a["iou"], 3)} if "mapa_som" in a else {})}
                           for a in sorted(lista, key=lambda a: (-a["di"], a["dj"]))]}
    with open(os.path.join(args.pasta_saida, nome + ".json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)
    print("Figura e resultados salvos em", args.pasta_saida)
