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
from datetime import datetime, timedelta
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import comparacaoGrade
import enchenteSOM
import sentinelUtils


# Evento: Pelotas - RS, cenas de antes e de depois
nome = "pelotas_expandido"
titulo_evento = "Pelotas, Brasil - evento em 26/05/2024"
centro_lon, centro_lat = -52.3265, -31.7725
data_antes = "2024-02-01"
data_depois = "2024-05-26"

# Parâmetros da expansão
meia_largura_graus = 0.025    # Cada área tem ~5,5 km de lado
raio_max = 6                  # No máximo 6 áreas do centro em cada direção (grade de até 13x13)
limiar_expansao_km2 = 0.5     # Alagamento (SOM) mínimo de uma área para expandir para as vizinhas
nuvens_max_area = 20          # % máxima de nuvens ou sem dado para processar uma área

# Parâmetros de ajuste do SOM e do baseline
patch_size = 1
som_size = 12
epochs = 3
dataset_percentage = 0.2
limiar_agua = 0


# Retângulo da área (di, dj): di linhas ao norte e dj colunas a leste do centro
def bbox_area(di, dj):
    return sentinelUtils.criar_bbox(centro_lon + dj * 2 * meia_largura_graus,
                                    centro_lat + di * 2 * meia_largura_graus, meia_largura_graus)

def nome_area(di, dj):
    return "%s%d %s%d" % ("N" if di >= 0 else "S", abs(di), "L" if dj >= 0 else "O", abs(dj))

# Pares (antes, depois) das duas datas, um por tile que tem as duas cenas
def pares_por_tile(bbox):
    def por_tile(data):
        dia = datetime.strptime(data, "%Y-%m-%d")
        return {item.properties["s2:mgrs_tile"]: item.id
                for item in sentinelUtils.buscar_cenas(bbox, dia, dia + timedelta(days=1))
                if item.datetime.date() == dia.date()}
    antes, depois = por_tile(data_antes), por_tile(data_depois)
    return [(antes[t], depois[t]) for t in sorted(antes) if t in depois]

# Processa uma área: escolhe o tile com menos pixels inválidos nas duas datas
# e aplica os dois métodos. As cenas são buscadas de novo aqui, para que as
# URLs assinadas não expirem durante processamentos longos
def processar(args):
    di, dj, pares, pasta_cache = args
    bbox = bbox_area(di, dj)
    melhor = None
    for id_antes, id_depois in pares:
        item_antes, item_depois = sentinelUtils.buscar_item(id_antes), sentinelUtils.buscar_item(id_depois)
        invalida = max(sentinelUtils.fracao_invalida(item_antes, bbox), sentinelUtils.fracao_invalida(item_depois, bbox))
        if melhor is None or invalida < melhor[0]:
            melhor = (invalida, item_antes, item_depois)
    _, item_antes, item_depois = melhor
    area = comparacaoGrade.processar_area(
        item_antes, item_depois, bbox, nome_area(di, dj), os.path.join(pasta_cache, "area_%d_%d.npz" % (di, dj)),
        nuvens_max_area, patch_size, som_size, epochs, dataset_percentage, limiar_agua)
    area.update(di=di, dj=dj, tile=item_depois.properties["s2:mgrs_tile"])
    return area


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explora as áreas vizinhas com alagamento")
    parser.add_argument("--pasta-saida", default=os.path.join("resultados", "expansao"))
    parser.add_argument("--pasta-cache", default="cache_explorar")
    parser.add_argument("--processos", type=int, default=4)
    parser.add_argument("--raio-max", type=int, default=raio_max, help="Distância máxima, em áreas, a partir do centro")
    args = parser.parse_args()
    raio_max = args.raio_max
    os.makedirs(args.pasta_saida, exist_ok=True)
    pasta_cache = os.path.join(args.pasta_cache, nome)
    os.makedirs(pasta_cache, exist_ok=True)

    regiao = sentinelUtils.criar_bbox(centro_lon, centro_lat, (2 * raio_max + 1) * meia_largura_graus)
    pares = pares_por_tile(regiao)
    print("Pares de cenas por tile:", pares, flush=True)

    areas = {}
    fila = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
    onda = 0
    with Pool(args.processos) as pool:
        while fila:
            print("\n#### Onda %d: %d áreas" % (onda, len(fila)), flush=True)
            for area in pool.map(processar, [(di, dj, pares, pasta_cache) for di, dj in fila]):
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
    fig = comparacaoGrade.figura_comparacao(lista, item_antes, item_depois, centro_lat, limiar_agua, titulo,
                                            os.path.join(args.pasta_saida, nome + ".png"), dpi=110,
                                            mostrar_nomes=False)
    plt.close(fig)

    resultado = {"nome": nome, "data_antes": data_antes, "data_depois": data_depois, "raio_max": raio_max,
                 "limiar_expansao_km2": limiar_expansao_km2, "totais": t,
                 "areas": [{"area": a["nome"], "di": a["di"], "dj": a["dj"], "tile": a["tile"], "bbox": a["bbox"],
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
