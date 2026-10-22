# -*- coding: utf-8 -*-
"""
Processa os eventos de inundação de eventos_inundacao.csv: para cada evento,
monta uma grade de áreas em torno da coordenada, escolhe automaticamente as
cenas Sentinel-2 de antes e de depois, aplica o SOM bi-temporal e o limiar
direto de NDWI em cada área e salva a figura de comparação em
<pasta_resultados>/<evento>.png. Os totais de cada evento vão para
<pasta_resultados>/<evento>.json e, juntos, para <pasta_resultados>/resumo.csv.

Escolha das cenas (sentinelUtils.selecionar_cenas): a de depois é a de menor
fração de nuvens/sem dado dentro da grade entre 1 dia antes e 5 dias depois da
data do evento (preferindo a data exata); a de antes é do mesmo tile, entre 180
e 30 dias antes do evento, com menos nuvens dentro da grade (preferindo as mais
recentes).

Uso:
    python processarEventos.py                      # todos os eventos
    python processarEventos.py --eventos Pelotas "Porto Alegre"
    python processarEventos.py --worker 0 --n-workers 3   # divide os eventos entre processos
    python processarEventos.py --resumo             # só refaz o resumo.csv

Eventos que já têm a figura são pulados (use --refazer para processar de novo).
O processamento e a figura estão em comparacaoGrade.py.
"""

import argparse
import csv
import glob
import json
import os
import re
import traceback
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import comparacaoGrade
import enchenteSOM
import sentinelUtils


# Parâmetros da grade (os mesmos de exemploComparacaoNDWI.py)
meia_largura_graus = 0.025   # Cada área tem ~5,5 km de lado
n_linhas, n_colunas = 3, 3   # Grade centrada na coordenada do evento
nuvens_max_area = 20         # % máxima de nuvens ou sem dado para processar uma área

# Parâmetros de ajuste do SOM
patch_size = 1
som_size = 12
epochs = 3
dataset_percentage = 0.2

limiar_agua = 0  # NDWI acima disso indica água, nos dois métodos


# Nome de arquivo sem acentos nem espaços (ex.: "Restinga Sêca" -> "restinga_seca")
def nome_arquivo(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")

def ler_eventos(arquivo):
    with open(arquivo, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def processar_evento(evento, pasta_resultados, pasta_cache):
    nome = nome_arquivo(evento["localidade"])
    lat, lon = float(evento["latitude"]), float(evento["longitude"])
    resumo = {"evento": nome, "localidade": evento["localidade"], "pais": evento["pais"], "tipo": evento["tipo"],
              "latitude": lat, "longitude": lon, "data_evento": evento["data"]}
    print("\n######## %s, %s - %s" % (evento["localidade"], evento["pais"], evento["data"]), flush=True)
    try:
        bbox = enchenteSOM.bbox_grade(lon, lat, meia_largura_graus, n_linhas, n_colunas)
        item_antes, item_depois, invalida_antes, invalida_depois = sentinelUtils.selecionar_cenas(bbox, evento["data"])
        print("Antes: %s (%.0f%% de nuvens/sem dado na grade)" % (item_antes.id, 100 * invalida_antes))
        print("Depois: %s (%.0f%% de nuvens/sem dado na grade)" % (item_depois.id, 100 * invalida_depois), flush=True)
        resumo.update(cena_antes=item_antes.id, cena_depois=item_depois.id,
                      data_antes=item_antes.datetime.strftime("%Y-%m-%d"),
                      data_depois=item_depois.datetime.strftime("%Y-%m-%d"),
                      pct_invalido_antes=round(100 * invalida_antes, 1),
                      pct_invalido_depois=round(100 * invalida_depois, 1))

        areas_grade = comparacaoGrade.processar_grade(
            item_antes, item_depois, lon, lat, meia_largura_graus, n_linhas, n_colunas,
            os.path.join(pasta_cache, nome), nuvens_max_area, patch_size, som_size, epochs,
            dataset_percentage, limiar_agua)
        resumo.update({k: round(v, 3) if isinstance(v, float) else v
                       for k, v in comparacaoGrade.totais(areas_grade).items()})

        titulo = ("%s, %s (%s) - evento em %s: SOM bi-temporal x limiar direto de NDWI (NDWI > %g)"
                  % (evento["localidade"], evento["pais"], evento["tipo"],
                     "/".join(reversed(evento["data"].split("-"))), limiar_agua))
        fig = comparacaoGrade.figura_comparacao(areas_grade, item_antes, item_depois, lat, limiar_agua, titulo,
                                                os.path.join(pasta_resultados, nome + ".png"), dpi=80)
        plt.close(fig)
        resumo["status"] = "ok" if resumo["areas_processadas"] > 0 else "sem áreas processadas (nuvens)"
    except Exception as erro:
        traceback.print_exc()
        resumo["status"] = "erro: %s" % erro
    print("Status:", resumo["status"], flush=True)
    with open(os.path.join(pasta_resultados, nome + ".json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=1)
    return resumo

# Junta os .json de todos os eventos em resumo.csv, na ordem da tabela de eventos
def escrever_resumo(pasta_resultados, eventos):
    ordem = {nome_arquivo(e["localidade"]): k for k, e in enumerate(eventos)}
    resumos = []
    for arquivo in glob.glob(os.path.join(pasta_resultados, "*.json")):
        with open(arquivo, encoding="utf-8") as f:
            resumos.append(json.load(f))
    resumos.sort(key=lambda r: ordem.get(r["evento"], len(ordem)))
    colunas = []
    for r in resumos:
        colunas += [c for c in r if c not in colunas]
    with open(os.path.join(pasta_resultados, "resumo.csv"), "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(resumos)
    print("\nResumo de %d eventos salvo em %s" % (len(resumos), os.path.join(pasta_resultados, "resumo.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa os eventos de inundação com o SOM bi-temporal e o NDWI")
    parser.add_argument("--arquivo-eventos", default="eventos_inundacao.csv")
    parser.add_argument("--eventos", nargs="*", help="Localidades a processar (padrão: todas)")
    parser.add_argument("--pasta-resultados", default="resultados")
    parser.add_argument("--pasta-cache", default="cache_eventos")
    parser.add_argument("--worker", type=int, default=0, help="Índice deste processo")
    parser.add_argument("--n-workers", type=int, default=1, help="Número de processos dividindo os eventos")
    parser.add_argument("--refazer", action="store_true", help="Processa de novo eventos que já têm figura")
    parser.add_argument("--resumo", action="store_true", help="Só refaz o resumo.csv")
    args = parser.parse_args()

    os.makedirs(args.pasta_resultados, exist_ok=True)
    eventos = ler_eventos(args.arquivo_eventos)
    if not args.resumo:
        selecionados = [e for e in eventos if not args.eventos or e["localidade"] in args.eventos]
        selecionados = selecionados[args.worker::args.n_workers]
        for evento in selecionados:
            figura = os.path.join(args.pasta_resultados, nome_arquivo(evento["localidade"]) + ".png")
            if os.path.exists(figura) and not args.refazer:
                print("Já processado:", evento["localidade"])
                continue
            processar_evento(evento, args.pasta_resultados, args.pasta_cache)
    escrever_resumo(args.pasta_resultados, eventos)
