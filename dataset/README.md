# Dataset de alagamentos

Imagens Sentinel-2 L2A (Microsoft Planetary Computer) de áreas alagadas e das
mesmas áreas antes do alagamento. Cada evento tem uma pasta, gerada por
`criarDataset.py` a partir das áreas exploradas por `explorarAlagamento.py`
(só as áreas com nuvens ou sem dado abaixo de 20% em ambas as datas).

## Eventos

| Pasta | Evento | Antes | Enchente | Áreas |
|---|---|---|---|---|
| `porto_alegre/` | Porto Alegre e vale do Jacuí - RS, maio de 2024 | 21/04/2024 | 06/05/2024 | 426 |
| `pelotas/` | Pelotas, Canal São Gonçalo e Rio Grande - RS, maio de 2024 | 01/02/2024 | 26/05/2024 (31/05/2024 em 36 áreas com nuvens em 26/05) | 193 |
| `kaoh_chen/` | Kaoh Chen e planície do Mekong - Camboja, julho de 2017 | 21/04/2017 | 30/07/2017 | 147 |
| `sa_khut/` | Sa Khut e vale do rio Mun - Tailândia, agosto de 2017 | 03/06/2017 | 02/08/2017 | 64 |
| `assuncao/` | Assunção e rio Paraguai - Paraguai, janeiro de 2016 | 22/10/2015 | 13/01/2016 | 95 |
| `restinga_seca/` | Restinga Sêca, alto Jacuí e Vacacaí - RS, maio de 2024 (só áreas fora de `porto_alegre/`) | 21/04/2024 | 06/05/2024 | 121 |
| `phrai_bueng/` | Phrai Bueng e várzea do rio Mun - Tailândia, agosto de 2017 | 05/03/2017 | 02/08/2017 | 46 |

## Arquivos de cada área

Cada área tem ~5,5 km de lado (0,05° x 0,05°). Os nomes seguem o padrão
`<evento>_<area>_lat<lat>_lon<lon>_...`, onde `<area>` é a posição na grade em
relação ao centro do evento (ex.: `N03_O12` = 3 áreas ao norte e 12 a oeste) e
`lat`/`lon` são as coordenadas do centro da área.

| Arquivo | Conteúdo |
|---|---|
| `..._antes_<data>_bruto.tif` | GeoTIFF, 5 bandas uint16: B02 (azul), B03 (verde), B04 (vermelho), B08 (infravermelho próximo), B11 (infravermelho de ondas curtas, reamostrada de 20 m). Reflectância de superfície = valor / 10000. Projeção UTM do tile, pixels de ~10 m. |
| `..._enchente_<data>_bruto.tif` | O mesmo, na data da enchente. As duas imagens de uma área estão na mesma grade de pixels. |
| `..._antes_<data>_rgb.png` e `..._enchente_<data>_rgb.png` | Composição em cor verdadeira (B04, B03, B02), reflectância 0-0,15 escalada para 0-255. |
| `..._mascara_invalidos.tif` | 1 = nuvem, sombra ou sem dado em qualquer uma das datas (banda SCL); 0 = válido. |
| `indice.csv` | Uma linha por imagem: área, coordenadas, data, cena e tile Sentinel-2, CRS, % de pixels inválidos, nomes dos arquivos e o alagamento e a água permanente estimados pelo SOM bi-temporal (km²). |

Os identificadores das cenas também ficam nas tags de cada GeoTIFF.

Uma área que cobre mais de 50% de uma área de outra pasta com as mesmas datas
não é salva de novo (ex.: Restinga Sêca e Porto Alegre usam as mesmas cenas, e
as 148 áreas de Restinga Sêca que já estavam em `porto_alegre/` foram puladas).

## Dados brutos fora do git

Os GeoTIFF brutos (`*_bruto.tif`, ~1,6 GB para Porto Alegre e ~0,7 GB para Pelotas e ~0,6 GB para Kaoh Chen, ~0,3 GB para Sa Khut e ~0,4 GB para Assunção, ~0,4 GB para Restinga Sêca e ~0,2 GB para Phrai Bueng) não são versionados
no git, para o repositório não ficar pesado. Estão no git as imagens RGB, as
máscaras e o `indice.csv`. Para gerar os brutos de novo:

    python explorarAlagamento.py --evento "Porto Alegre" --raio-max 40   # baixa as cenas e processa as áreas
    python criarDataset.py --evento "Porto Alegre"

    python explorarAlagamento.py --raio-max 40 --incluir -32.15 -31.75 -52.35 -51.95 --datas-depois-alternativas 2024-05-31
    python criarDataset.py --evento "Pelotas"

    python explorarAlagamento.py --evento "Kaoh Chen" --raio-max 30
    python criarDataset.py --evento "Kaoh Chen"

    python explorarAlagamento.py --evento "Sa Khut" --raio-max 15
    python criarDataset.py --evento "Sa Khut"

    python explorarAlagamento.py --evento "Assunção" --raio-max 15
    python criarDataset.py --evento "Assunção"

    python explorarAlagamento.py --evento "Restinga Sêca" --raio-max 25
    python criarDataset.py --evento "Restinga Sêca"   # depois de porto_alegre, para pular as áreas repetidas

    python explorarAlagamento.py --evento "Phrai Bueng" --raio-max 15
    python criarDataset.py --evento "Phrai Bueng"

`explorarAlagamento.py` guarda as cenas lidas em `cache_explorar/`, que
`criarDataset.py` usa para escrever o dataset. As cenas usadas em cada área estão
no `indice.csv`, então a regeneração reproduz exatamente as mesmas imagens.
