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
        Composição em cor verdadeira (B04, B03, B02).
    <evento>_<area>_lat<lat>_lon<lon>_mascara_invalidos.tif
        Pixels com nuvem, sombra ou sem dado em qualquer uma das datas (1) e
        válidos (0).
<area> é a posição na grade em relação ao centro do evento (ex.: N03_O12 = 3
áreas ao norte e 12 a oeste) e lat/lon são as coordenadas do centro da área.
O índice de todos os arquivos fica em dataset/<evento>/indice.csv.

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

    linhas = []
    for arquivo_resultado in sorted(glob.glob(os.path.join(pasta_cache, "resultado_*.pkl"))):
        with open(arquivo_resultado, "rb") as f:
            area = pickle.load(f)
        arquivo_cache = os.path.join(pasta_cache, "area_%d_%d.npz" % (area["di"], area["dj"]))
        if "mapa_som" not in area or not os.path.exists(arquivo_cache):
            continue  # Área pulada (nuvens ou sem dado): não tem dados brutos
        dados = np.load(arquivo_cache)
        id_antes, id_depois = [str(c) for c in dados["cenas"]]
        item_antes, item_depois = item(id_antes), item(id_depois)

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

        for estado, chave, it in [("antes", "refl_antes", item_antes), ("enchente", "refl_depois", item_depois)]:
            data = it.datetime.strftime("%Y-%m-%d")
            base = "%s_%s_%s" % (prefixo, estado, data)
            refl = dados[chave]
            bruto = np.round(np.clip(refl, 0, 1) * ESCALA_REFLECTANCIA).astype(np.uint16).transpose(2, 0, 1)
            salvar_geotiff(os.path.join(pasta, base + "_bruto.tif"), bruto, crs, transform, enchenteSOM.BANDAS,
                           {"cena": it.id, "data": data, "estado": estado,
                            "escala": "reflectancia = valor / %d" % ESCALA_REFLECTANCIA})
            rgb = (enchenteSOM.cor_verdadeira(refl) * 255).astype(np.uint8)
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
