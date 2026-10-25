# -*- coding: utf-8 -*-
"""
Cria o dataset de alagamentos a partir das áreas exploradas por
explorarAlagamento.py: para cada área processada, salva os dados brutos e a
imagem RGB da cena de antes e da cena da enchente.

Arquivos de cada área, em dataset/<evento>/:
    <evento>_<area>_lat<lat>_lon<lon>_<estado>_<data>_bruto.tif
        GeoTIFF com as bandas B02, B03, B04, B08 e B11 (reflectância de
        superfície Sentinel-2 L2A x 10000, uint16), na projeção UTM do tile,
        com pixels de 10 m. <estado> é "antes" ou "enchente".
    <evento>_<area>_lat<lat>_lon<lon>_<estado>_<data>_rgb.png
        Composição em cor verdadeira (B04, B03, B02); escala fixa (0-0,15) ou, com
        --rgb-contraste auto, percentis 2-98% da área nas duas datas.
    <evento>_<area>_lat<lat>_lon<lon>_mascara_invalidos.tif
        Pixels com nuvem, sombra ou sem dado em qualquer uma das datas (1) e
        válidos (0).
<area> é a posição na grade em relação ao centro do evento (ex.: N03_O12 = 3
áreas ao norte e 12 a oeste) e lat/lon são as coordenadas do centro da área.
O índice de todos os arquivos fica em dataset/<evento>/indice.csv.
Áreas que já estão em outra pasta do dataset com as mesmas datas (sobreposição
acima de --sobreposicao-max) são puladas, para não duplicar imagens.

Uso:
    python criarDataset.py --evento "Porto Alegre" --pasta-cache <cache de explorarAlagamento.py>
"""

import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

import enchenteSOM
import processarEventos
import sentinelUtils


ESCALA_REFLECTANCIA = 10000


def nome_area(di, dj):
    return "%s%02d_%s%02d" % ("N" if di >= 0 else "S", abs(di), "L" if dj >= 0 else "O", abs(dj))

# Código EPSG da projeção de uma cena (ex.: 32722 para UTM 22S)
def epsg_da_cena(item):
    codigo = item.properties.get("proj:epsg") or item.properties.get("proj:code")
    return int(str(codigo).split(":")[-1])

def salvar_geotiff(arquivo, dados, crs, transform, descricoes, extras):
    with rasterio.open(arquivo, "w", driver="GTiff", height=dados.shape[1], width=dados.shape[2],
                       count=dados.shape[0], dtype=dados.dtype, crs=crs, transform=transform,
                       compress="deflate", predictor=2) as dst:
        dst.write(dados)
        for k, d in enumerate(descricoes):
            dst.set_band_description(k + 1, d)
        dst.update_tags(**extras)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cria o dataset de alagamentos de um evento")
    parser.add_argument("--evento", default="Porto Alegre", help="Localidade de eventos_inundacao.csv")
    parser.add_argument("--pasta-cache", default="cache_explorar", help="Cache usado por explorarAlagamento.py")
    parser.add_argument("--pasta-dataset", default="dataset")
    parser.add_argument("--rgb-contraste", choices=["fixo", "auto"], default="fixo",
                        help="fixo: reflectância 0-0,15 vira 0-255; auto: percentis 2-98%% de cada área (as duas "
                             "datas com a mesma escala), melhor em regiões claras como deserto")
    parser.add_argument("--sobreposicao-max", type=float, default=0.5,
                        help="Pula áreas que cobrem mais que essa fração de uma área já existente em outra pasta "
                             "do dataset com as mesmas datas (evita imagens duplicadas)")
    args = parser.parse_args()

    evento = processarEventos.nome_arquivo(args.evento)
    pasta_cache = os.path.join(args.pasta_cache, evento + "_expandido")
    pasta = os.path.join(args.pasta_dataset, evento)
    os.makedirs(pasta, exist_ok=True)

    itens = {}  # Cenas já buscadas, por id
    def item(id_cena):
        if id_cena not in itens:
            itens[id_cena] = sentinelUtils.buscar_item(id_cena)
        return itens[id_cena]

    # Áreas já salvas em outras pastas do dataset: (bbox, data de antes, data da enchente)
    existentes = []
    for indice in glob.glob(os.path.join(args.pasta_dataset, "*", "indice.csv")):
        if os.path.dirname(indice) == pasta:
            continue
        with open(indice, encoding="utf-8") as f:
            por_area = {}
            for l in csv.DictReader(f):
                por_area.setdefault(l["arquivo_mascara"], {})[l["estado"]] = l
        for d in por_area.values():
            l = d["enchente"]
            existentes.append(([float(l[k]) for k in ("lon_min", "lat_min", "lon_max", "lat_max")],
                               d["antes"]["data"], l["data"], l["evento"]))

    def sobreposta(bbox, data_antes, data_enchente):
        for b, da, de, ev in existentes:
            if (da, de) != (data_antes, data_enchente):
                continue
            larg = min(bbox[2], b[2]) - max(bbox[0], b[0])
            alt = min(bbox[3], b[3]) - max(bbox[1], b[1])
            if larg > 0 and alt > 0 and larg * alt > args.sobreposicao_max * (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]):
                return ev
        return None

    linhas = []
    puladas_sobreposicao = {}
    for arquivo_resultado in sorted(glob.glob(os.path.join(pasta_cache, "resultado_*.pkl"))):
        with open(arquivo_resultado, "rb") as f:
            area = pickle.load(f)
        arquivo_cache = os.path.join(pasta_cache, "area_%d_%d.npz" % (area["di"], area["dj"]))
        if "mapa_som" not in area or not os.path.exists(arquivo_cache):
            continue  # Área pulada (nuvens ou sem dado): não tem dados brutos
        dados = np.load(arquivo_cache)
        id_antes, id_depois = [str(c) for c in dados["cenas"]]
        item_antes, item_depois = item(id_antes), item(id_depois)
        ev = sobreposta(area["bbox"], item_antes.datetime.strftime("%Y-%m-%d"), item_depois.datetime.strftime("%Y-%m-%d"))
        if ev:
            puladas_sobreposicao[ev] = puladas_sobreposicao.get(ev, 0) + 1
            continue

        # Georreferência: a área foi lida do retângulo da área, reprojetado para o UTM do tile
        crs = "EPSG:%d" % epsg_da_cena(item_depois)
        altura, largura = dados["nuvens"].shape
        transform = from_bounds(*transform_bounds("EPSG:4326", crs, *area["bbox"]), largura, altura)

        lon_c = (area["bbox"][0] + area["bbox"][2]) / 2
        lat_c = (area["bbox"][1] + area["bbox"][3]) / 2
        prefixo = "%s_%s_lat%.3f_lon%.3f" % (evento, nome_area(area["di"], area["dj"]), lat_c, lon_c)

        arquivo_mascara = prefixo + "_mascara_invalidos.tif"
        salvar_geotiff(os.path.join(pasta, arquivo_mascara), dados["nuvens"][None].astype(np.uint8), crs, transform,
                       ["invalido (nuvem, sombra ou sem dado em qualquer data)"],
                       {"cena_antes": id_antes, "cena_enchente": id_depois})

        if args.rgb_contraste == "auto":
            rgb_todos = np.concatenate([dados[c][:, :, [2, 1, 0]][~dados["nuvens"]] for c in ("refl_antes", "refl_depois")])
            rgb_lo, rgb_hi = np.percentile(rgb_todos, [2, 98]) if len(rgb_todos) else (0, 0.15)
        for estado, chave, it in [("antes", "refl_antes", item_antes), ("enchente", "refl_depois", item_depois)]:
            data = it.datetime.strftime("%Y-%m-%d")
            base = "%s_%s_%s" % (prefixo, estado, data)
            refl = dados[chave]
            bruto = np.round(np.clip(refl, 0, 1) * ESCALA_REFLECTANCIA).astype(np.uint16).transpose(2, 0, 1)
            salvar_geotiff(os.path.join(pasta, base + "_bruto.tif"), bruto, crs, transform, enchenteSOM.BANDAS,
                           {"cena": it.id, "data": data, "estado": estado,
                            "escala": "reflectancia = valor / %d" % ESCALA_REFLECTANCIA})
            if args.rgb_contraste == "auto":
                rgb = (np.clip((refl[:, :, [2, 1, 0]] - rgb_lo) / max(rgb_hi - rgb_lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            else:
                rgb = (enchenteSOM.cor_verdadeira(refl, 0.15) * 255).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(pasta, base + "_rgb.png"))
            linhas.append({"evento": args.evento, "area": nome_area(area["di"], area["dj"]),
                           "linha_grade": area["di"], "coluna_grade": area["dj"],
                           "lat_centro": round(lat_c, 5), "lon_centro": round(lon_c, 5),
                           "lon_min": area["bbox"][0], "lat_min": area["bbox"][1],
                           "lon_max": area["bbox"][2], "lat_max": area["bbox"][3],
                           "estado": estado, "data": data, "cena": it.id, "tile": it.properties["s2:mgrs_tile"],
                           "crs": crs, "altura": altura, "largura": largura,
                           "pct_invalido": round(float(dados["nuvens"].mean()) * 100, 2),
                           "arquivo_bruto": base + "_bruto.tif", "arquivo_rgb": base + "_rgb.png",
                           "arquivo_mascara": arquivo_mascara,
                           "som_alagamento_km2": round(float(area["areas_som"][2]), 3),
                           "som_agua_permanente_km2": round(float(area["areas_som"][1]), 3)})
        print(prefixo, flush=True)

    with open(os.path.join(pasta, "indice.csv"), "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)
    print("\n%d áreas, %d imagens salvas em %s" % (len(linhas) // 2, len(linhas), pasta))
    for ev, n in puladas_sobreposicao.items():
        print("%d áreas puladas por já estarem na pasta de %s (mesmas datas)" % (n, ev))
