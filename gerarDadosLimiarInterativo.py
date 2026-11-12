# -*- coding: utf-8 -*-
"""
Gera os dados da página interativa do limiar do SOM (resultados/sen1floods11/
limiar_interativo.html) para um recorte do Sen1Floods11: o SOM pré-treinado
(uma imagem balanceada de cada evento, normalização fixa) segmenta o recorte sem
classificar as unidades, e são guardados a unidade de cada pixel, o MNDWI médio
de cada unidade nas duas datas, a referência (rótulo à mão + água permanente do
JRC), as imagens em cor verdadeira e o MNDWI de cada pixel, para a página
aplicar o limiar no navegador.

Uso:
    python gerarDadosLimiarInterativo.py <pasta_cache> <recorte> <saida.json>
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
from experimentoAreaAlagada import ler_jrc
from experimentoPreTreinoEvento import escolher_pre_treino
from experimentoPreTreinoMultiEvento import AGUA_MAX, AGUA_MIN
from experimentoPreTreinoSOM import pre_treinar
from variacaoUnidadesSOM import ler_recorte


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def jpg(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


C, recorte, saida = sys.argv[1:4]
por_evento = defaultdict(list)
for f in glob.glob(C + "/resultados/*.json"):
    r = json.load(open(f))
    if r["status"] == "ok":
        por_evento[r["pais"]].append(r)
imagens = sorted(e["recorte"] for e in (escolher_pre_treino(l) for l in por_evento.values())
                 if AGUA_MIN <= e["pct_agua_rotulo"] <= AGUA_MAX)
SOM, _ = pre_treinar(imagens, "fixa", 0, C)

r = json.load(open(C + "/resultados/" + recorte + ".json"))
rotulo, antes, depois, invalidos = ler_recorte(r, C)
jrc = ler_jrc(recorte, C)
validos = (rotulo >= 0) & ~invalidos
entrada, _ = es.montar_entrada(antes, depois, normalizacao="fixa")
result = es.segmentar(SOM, entrada, 1)
m_antes, m_depois = es.indice_agua(antes), es.indice_agua(depois)

unidades = []
for u in range(es.SOM_SIZE):
    m = (result == u) & ~invalidos
    unidades.append({"u": u, "pixels": int(m.sum()),
                     "mndwi_antes": round(float(m_antes[m].mean()), 4) if m.any() else None,
                     "mndwi_evento": round(float(m_depois[m].mean()), 4) if m.any() else None})
ref = np.zeros(rotulo.shape, np.uint8)  # 0 não água, 1 permanente, 2 alagamento, 3 sem dado
ref[(rotulo == 1) & jrc] = 1
ref[(rotulo == 1) & ~jrc] = 2
ref[~validos] = 3
lo, hi = np.percentile(np.concatenate([x[:, :, [2, 1, 0]][validos] for x in (antes, depois)]), [2, 98])
rgb = [(np.clip((x[:, :, [2, 1, 0]] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8) for x in (antes, depois)]
q = lambda m: np.clip(np.round((m + 1) * 127.5), 0, 255).astype(np.uint8)
mndwi_png = np.dstack([q(m_antes), q(m_depois), np.zeros_like(q(m_antes))])

dados = {"recorte": recorte, "pais": r["pais"], "data_antes": r["data_antes"], "data_evento": r["data_depois"],
         "largura": int(rotulo.shape[1]), "altura": int(rotulo.shape[0]), "unidades": unidades,
         "imagem_antes": jpg(rgb[0]), "imagem_evento": jpg(rgb[1]), "unidade_pixel": png(result.astype(np.uint8)),
         "referencia": png(ref), "mndwi_pixel": png(mndwi_png),
         "pct_alagado_ref": round(100 * float((ref == 2)[validos].mean()), 2),
         "pct_permanente_ref": round(100 * float((ref == 1)[validos].mean()), 2)}
json.dump(dados, open(saida, "w"))
print(recorte, "alagamento %.1f%%, permanente %.1f%%" % (dados["pct_alagado_ref"], dados["pct_permanente_ref"]))
for u in sorted(unidades, key=lambda x: -(x["mndwi_evento"] or -9))[:8]:
    print(u)
