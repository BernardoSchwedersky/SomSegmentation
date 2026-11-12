# -*- coding: utf-8 -*-
"""
Gera os dados da página interativa do limiar de posição do SOM
(resultados/sen1floods11/limiar_posicao_interativo.html) para um recorte do
Sen1Floods11.

O SOM (um único SOM 1D para as duas datas, experimentoLimiarPosicao.py) é
pré-treinado com uma imagem balanceada de cada evento e aplicado ao recorte
sem classificar as unidades: são guardadas a unidade de cada pixel antes e no
evento, a cor do protótipo de cada unidade, a referência (rótulo à mão + água
permanente do JRC), as imagens em cor verdadeira, o limiar k e o lado da água
escolhidos no pré-treino, e o IoU do MNDWI por pixel (limiar 0) para comparar.

Uso:
    python gerarDadosLimiarPosicao.py <pasta_cache> <recorte> <saida.json>
"""

import base64
import glob
import io
import json
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

import enchenteSOM as es
from avaliarSen1Floods11 import contar, metricas
from experimentoAreaAlagada import ler_jrc
from experimentoLimiarPosicao import pre_treinar_posicao, resultados_por_data
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from variacaoUnidadesSOM import ler_recorte


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def jpg(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


if __name__ == "__main__":
    C, recorte, saida = sys.argv[1:4]
    por_evento = defaultdict(list)
    for f in glob.glob(C + "/resultados/*.json"):
        r = json.load(open(f))
        if r["status"] == "ok":
            por_evento[r["pais"]].append(r)
    imagens = sorted(e["recorte"] for e in (escolher_pre_treino(l) for l in por_evento.values())
                     if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX)
    m = pre_treinar_posicao(imagens, 0, C)

    r = json.load(open(C + "/resultados/" + recorte + ".json"))
    rotulo, antes, depois, invalidos = ler_recorte(r, C)
    jrc = ler_jrc(recorte, C)
    validos = (rotulo >= 0) & ~invalidos
    ra, rd = resultados_por_data(m["SOM"], antes, depois)

    ref = np.zeros(rotulo.shape, np.uint8)  # 0 não água, 1 permanente, 2 alagamento, 3 sem dado
    ref[(rotulo == 1) & jrc] = 1
    ref[(rotulo == 1) & ~jrc] = 2
    ref[~validos] = 3
    lo, hi = np.percentile(np.concatenate([x[:, :, [2, 1, 0]][validos] for x in (antes, depois)]), [2, 98])
    cor = lambda refl: np.clip((refl - lo) / (hi - lo), 0, 1) * 255
    rgb = [cor(x[:, :, [2, 1, 0]]).astype(np.uint8) for x in (antes, depois)]

    # Cor do protótipo: pixel central da janela 3x3 nas bandas B04, B03, B02, reflectância 0-0,2
    pesos = m["SOM"]["weights"][:, 0, :]
    centro = lambda b: pesos[:, es.BANDAS.index(b) * 9 + 4] / 255 * es.REFL_MAX_FIXA
    prot = (np.clip(np.stack([centro("B04"), centro("B03"), centro("B02")], axis=1) / 0.2, 0, 1) * 255).astype(int)
    unidades = [{"u": u, "pixels_antes": int(((ra == u) & ~invalidos).sum()),
                 "pixels_evento": int(((rd == u) & ~invalidos).sum()),
                 "cor": "#%02x%02x%02x" % tuple(prot[u])} for u in range(es.SOM_SIZE)]
    mndwi = es.mapa_indice(antes, depois, 0, "MNDWI")

    dados = {"recorte": recorte, "pais": r["pais"], "data_antes": r["data_antes"], "data_evento": r["data_depois"],
             "largura": int(rotulo.shape[1]), "altura": int(rotulo.shape[0]), "unidades": unidades,
             "k_pre_treino": m["k"], "agua_acima_pre_treino": bool(m["agua_acima"]),
             "imagem_antes": jpg(rgb[0]), "imagem_evento": jpg(rgb[1]),
             "unidade_antes": png(ra.astype(np.uint8)), "unidade_evento": png(rd.astype(np.uint8)),
             "referencia": png(ref),
             "iou_mndwi_pixel": round(metricas(contar(mndwi == 2, ref == 2, validos))["IoU"], 4),
             "pct_alagado_ref": round(100 * float((ref == 2)[validos].mean()), 2),
             "pct_permanente_ref": round(100 * float((ref == 1)[validos].mean()), 2)}
    json.dump(dados, open(saida, "w"))
    print(recorte, "k=%d água %s; MNDWI por pixel IoU %.3f" % (m["k"], "acima" if m["agua_acima"] else "abaixo",
                                                                dados["iou_mndwi_pixel"]))
    for u in unidades:
        print(u)
