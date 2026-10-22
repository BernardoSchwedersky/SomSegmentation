# -*- coding: utf-8 -*-
"""
Compara o mapeamento de enchente pelo SOM bi-temporal com o uso direto de um
índice espectral: o limiar de NDWI aplicado às imagens de antes e de depois.

No método direto, um pixel é água se NDWI > limiar_agua. Água nas duas datas é
água permanente e água só depois é área inundada. Esse método não tem
equivalente à classe "vegetação afetada" do SOM.

Usa a mesma grade, as mesmas cenas e a mesma pasta de cache de
exemploSentinelGrade.py: se a grade já foi processada, as imagens não são
baixadas de novo. A figura mostra os dois mapas, onde eles concordam e
discordam sobre a água e a área atingida em cada área da grade.

O processamento e a figura estão em comparacaoGrade.py, as etapas do método em
enchenteSOM.py e as funções de busca e leitura em sentinelUtils.py (ver
dependências lá). Para processar vários eventos, veja processarEventos.py.
"""

import matplotlib.pyplot as plt

import comparacaoGrade
import enchenteSOM
import sentinelUtils


# Parâmetros da grade e das imagens (iguais aos de exemploSentinelGrade.py)
centro_lon, centro_lat = -52.3265, -31.7725   # Centro da grade: Pelotas - RS
meia_largura_graus = 0.025                     # Cada área tem ~5,5 km de lado
n_linhas, n_colunas = 3, 3                     # Tamanho da grade (linha 0 = norte)
nuvens_max_area = 20                           # % máxima de nuvens para processar uma área

periodo_antes = "2024-03-07/2024-03-07"        # Cena antes da enchente
periodo_depois = "2024-05-26/2024-05-26"       # Data do evento
cobertura_nuvens_max = 10                      # Porcentagem máxima de nuvens na cena

pasta_cache = "cache_grade"                    # Mesmo cache de exemploSentinelGrade.py
arquivo_figura = "comparacao_som_ndwi.png"

# Parâmetros de ajuste do SOM
patch_size = 1
som_size = 12
epochs = 3
dataset_percentage = 0.2

limiar_agua = 0  # NDWI acima disso indica água, nos dois métodos


bbox_total = enchenteSOM.bbox_grade(centro_lon, centro_lat, meia_largura_graus, n_linhas, n_colunas)
item_antes = sentinelUtils.buscar_cena(bbox_total, periodo_antes, cobertura_nuvens_max)
item_depois = sentinelUtils.buscar_cena(bbox_total, periodo_depois, cobertura_nuvens_max)
print("Cenas:", item_antes.id, item_depois.id)

areas_grade = comparacaoGrade.processar_grade(item_antes, item_depois, centro_lon, centro_lat, meia_largura_graus,
                                              n_linhas, n_colunas, pasta_cache, nuvens_max_area, patch_size,
                                              som_size, epochs, dataset_percentage, limiar_agua)

comparacaoGrade.figura_comparacao(areas_grade, item_antes, item_depois, centro_lat, limiar_agua, arquivo=arquivo_figura)
print("\nFigura salva em", arquivo_figura)

t = comparacaoGrade.totais(areas_grade)
print("Área atingida nas %d áreas processadas: SOM %.2f km² (inundada + vegetação afetada), NDWI %.2f km²"
      % (t["areas_processadas"], t["som_atingida_km2"], t["ndwi_inundada_km2"]))

plt.show()
