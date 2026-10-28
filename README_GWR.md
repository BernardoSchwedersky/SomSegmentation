# GWR semi-supervisionado para segmentação online de vídeo aéreo

Extensão do `somSeg.py` com uma rede **Grow When Required** (Marsland, Shapiro & Nehmzow, 2002) para segmentação online de vídeo de UAV, com poucos rótulos.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `gwrSeg.py` | Implementação do GWR no mesmo estilo do `somSeg.py` (rede como dicionário) |
| `ruralscapes_gwr.py` | Experimento prequencial no Ruralscapes, comparando o GWR com o SOM do `somSeg.py` e com k-means online, além de um modo sintético |

## O que o `gwrSeg.py` faz

**GWR clássico:**
- crescimento por novidade (atividade baixa com o vencedor já habituado);
- habituação dos nós;
- envelhecimento e remoção de arestas e de nós isolados.

**Extensões semi-supervisionadas:**
- **Rótulos por nó:** cada nó guarda uma distribuição `P` e um contador de evidência `n`, atualizados pelas amostras rotuladas.
- **Propagação de rótulos** (`label_propagation`): os rótulos se espalham pelo grafo do próprio GWR, e nós com mais evidência ficam ancorados.
- **Crescimento guiado por conflito:** um nó confiante que erra repetidamente ganha um vizinho com o rótulo correto.
- **Reciclagem LRU:** com a rede cheia, o nó menos usado recentemente (de preferência sem rótulo) é substituído. Assim a rede continua se adaptando em voos longos.

**Saídas de inferência** (`predict_GWR`): rótulo, novidade (distância ao vencedor) e entropia do rótulo.

**Calibração** (`calibrate_dist_scale`): estima a escala de distância a partir dos dados, para que o limiar de atividade padrão (~exp(-1)) funcione sem ajuste fino. O parâmetro `n_protos` controla o tamanho típico da rede.

**Features** (`extract_features`):
- `patch`: o mesmo vetor do `prepare_dataset_multichanel`, porém vetorizado;
- `stats`: média e desvio padrão locais em Lab, mais gradiente.

## Uso

```bash
pip install numpy scipy scikit-image scikit-learn matplotlib pillow progress plotly

# Validação com sequência sintética (mudança de "estação" no meio)
python ruralscapes_gwr.py --synthetic

# Ruralscapes: rgb_frames/<video>_all/*.jpg + labels/<video>/seg_*.npz
# (ou máscaras PNG coloridas numa pasta com "label" no nome)
python ruralscapes_gwr.py --root /caminho/ruralscapes --videos DJI_0043 DJI_0051 \
    --label-frac 0.01 --out resultados_rural

# Classes agrupadas para pouso de emergência
python ruralscapes_gwr.py --root ... --classes landing

# UAVid: uavid_*/seqN/Images/*.png + uavid_*/seqN/Labels/*.png (paleta oficial, 8 classes)
python ruralscapes_gwr.py --dataset uavid --root /caminho/uavid --videos seq1 seq2 seq3

# Teste de sanidade: voo simulado (janela deslizante) sobre uma única imagem rotulada
python ruralscapes_gwr.py --dataset uavid4 --pan img.png mask.png
```

### Semantic Drone Dataset (TU Graz)

O SDD não é um vídeo: são 400 fotos nadir de 6000×4000 px. Mas o EXIF guarda o horário de captura, e o `prepare_sdd.py` usa essa informação para:

- ordenar as fotos na ordem real dos voos;
- reduzir a resolução;
- separar os voos por intervalos de mais de 30 minutos entre fotos.

O resultado são 5 voos em dois dias: 21/02/2017 (inverno, voos 00 e 01) e 11/06/2017 (verão, voos 02 a 04). A troca de estação é real.

```bash
python prepare_sdd.py --src semantic_drone_dataset --dst sdd_prep --width 1500
python ruralscapes_gwr.py --dataset sdd --root sdd_prep --classes landing --protos 10
```

As classes 0 (`unlabeled`) e 23 (`conflicting`) são ignoradas. No agrupamento `landing`, as classes ficam assim:

| Grupo | Classes do SDD |
|---|---|
| Pousável | paved-area, dirt, grass, gravel, ar-marker |
| Água | water, pool |
| Não pousável | todas as demais |

O SDD não tem céu, porque as fotos são nadir.

Resultado (400 fotos, 1% das amostras rotuladas, avaliação prequencial):

| Método | mIoU (pouso) | IoU água | mIoU (24 classes) | Nós | Inferência (ms) |
|---|---|---|---|---|---|
| GWR-semi (`--protos 10`) | 0,452 | 0,21 | — | 432 | 15 |
| GWR-semi (`--protos 30`) | 0,451 | 0,21 | 0,120 | 600 | 36 |
| k-means online | 0,395 | 0,07 | 0,095 | 100 | 7 |
| SOM (`somSeg`) | 0,333 | 0,04 | 0,062 | 100 | 5 |

Com `--protos 30`, a rede fica no limite de nós e passa a reciclar continuamente. Com `--protos 10`, ela estabiliza sem reciclar e é mais rápida, com o mesmo mIoU. Esse valor foi escolhido olhando os próprios dados de avaliação; num artigo, fixe-o usando só o primeiro voo.

### Banco de filtros convolucionais na entrada (`--features filters`)

A opção `--features filters` substitui as estatísticas locais de cor por um banco de filtros convolucionais no estilo dos *textons*. Cada resposta é retificada e agregada localmente com uma gaussiana. São 22 features:

| Grupo | Filtros | Features |
|---|---|---|
| Cor | L, a, b suavizados (σ = 1, 3, 8) | 9 |
| Gabor | Energia em quadratura, 6 orientações × λ = 4, 8, 16; máximo e média sobre as orientações | 6 |
| LoG | Laplaciano da gaussiana em módulo (σ = 1, 2, 4) | 3 |
| Gradiente | Magnitude do gradiente gaussiano (σ = 1, 3) | 2 |
| Desvio local | Desvio padrão local de L (σ = 3, 8) | 2 |

Detalhes:

- **Invariância à rotação:** como a câmera é nadir, a orientação da cena é arbitrária. Por isso o Gabor guarda só o máximo e a média sobre as orientações, e não a resposta de cada uma.
- **Normalização:** z-score e PCA (`--pca`) são ajustados uma única vez, com o primeiro frame e sem rótulos, e depois ficam fixos.

**Protocolo.** A configuração (`--pca 12 --protos 10`) foi escolhida usando só o voo_00. O teste usa os voos 01–04.

| Variante (validação, voo_00) | mIoU |
|---|---|
| stats (baseline) | 0,418 |
| filters | 0,428 |
| filters + PCA 6 | 0,432 |
| **filters + PCA 12** | **0,447** |
| filters + PCA 12, protos 5 / 20 | 0,411 / 0,414 |

Teste nos voos 01–04, com 3 sementes:

| GWR-semi | mIoU teste | Pousável | Não pousável | Água | Features | Inferência |
|---|---|---|---|---|---|---|
| stats | 0,452 ± 0,005 | 0,712 | 0,421 | 0,222 | 27 ms | 17 ms |
| filters + PCA 12 | **0,472 ± 0,010** | 0,711 | 0,444 | 0,261 | 230 ms | 31 ms |

- **Ganho:** +0,020 de mIoU, positivo nas três sementes (+0,013, +0,021, +0,027).
- **Qualitativo:** as segmentações ficam bem menos ruidosas, e objetos como a mesa e o carro passam a ser recortados inteiros.
- **Custo:** a extração de features fica cerca de 8× mais cara na CPU (2 núcleos, 480×320). Numa GPU embarcada, essas convoluções são baratas.
- **Comparação com os outros métodos** (semente 0, teste): com os mesmos filtros, o k-means também melhora (0,387 → 0,430) e o SOM piora (0,330 → 0,308). O GWR-semi continua o melhor (0,465).

### Redução gradual de resolução, no estilo do encoder da U-Net

**Encoder em pirâmide** (`--features pyramid --levels N`): em cada nível, o mesmo banco pequeno de 8 filtros é aplicado:

- L, a, b suavizados;
- energia de Gabor com λ = 4, máximo e média sobre as orientações;
- |LoG|;
- gradiente;
- desvio local.

Entre os níveis, a imagem passa por convolução gaussiana e redução 2×2 (`cv2.pyrDown`). Como os núcleos não crescem, o campo receptivo dobra a cada nível (λ efetivo de 4, 8, 16, 32... px), com custo 4× menor por nível. Os mapas voltam à resolução original e são concatenados, como as conexões de salto. Para visualizar o que cada nível captura, use `figura_piramide.py`.

**Saída em grade grossa** (`--out-factor f`): o GWR classifica só 1 a cada f pontos em cada direção, e as probabilidades e a novidade são interpoladas de volta à grade de avaliação.

**Validação** (voo_00, 3 sementes; mIoU):

| Variante | mIoU |
|---|---|
| filters (anterior) | 0,424 ± 0,021 |
| pirâmide 2 níveis | 0,421 ± 0,018 |
| pirâmide 4 níveis | 0,422 ± 0,026 |
| **pirâmide 6 níveis** | **0,448 ± 0,015** (escolhida) |

**Teste** (voos 01–04):

| GWR-semi | mIoU teste | Features | Inferência GWR |
|---|---|---|---|
| filters (anterior), 3 sementes | 0,472 ± 0,010 | 222 ms | 30 ms |
| pirâmide 6 níveis, 3 sementes | 0,447 ± 0,008 | 117 ms | 30 ms |
| pirâmide 6 níveis, `--out-factor 2` | 0,442 | 117 ms | 7,6 ms |
| pirâmide 6 níveis, `--out-factor 4` | 0,442 | 117 ms | 2,4 ms |
| *a posteriori:* pirâmide 4 níveis | 0,467 | 100 ms | 30 ms |
| *a posteriori:* filters + `--out-factor 4` | 0,483 | 236 ms | 2,2 ms |

Conclusões:

1. **A configuração escolhida na validação piorou no teste**, de forma consistente nas três sementes (−0,02 a −0,03). Os níveis 4 e 5 (30×20 e 15×10 px) descrevem a cena quase inteira e não generalizam entre voos com altitudes diferentes. O voo_00 favoreceu essa configuração porque quase não tem água. Num artigo, use validação com mais de um voo (por exemplo, *leave-one-flight-out*).
2. **Com 4 níveis, a pirâmide empata com o banco anterior** pela metade do custo de extração. Esse resultado é de uma semente e foi visto no teste.
3. **A saída em grade grossa reduz a inferência do GWR em 12×** (30 → 2,4 ms) sem perda de mIoU. A interpolação das probabilidades ainda suaviza o resultado.

### Protocolo "uma imagem" (`experimento_uma_imagem.py`)

**Regras:**

- Todos os modelos são treinados com **uma única imagem**, na qual só 1% dos pontos da grade (96 pontos) recebe rótulo (pousável / não pousável / água).
- Depois disso, **nenhum modelo recebe anotação**. As outras 399 imagens são classificadas usando só a própria imagem.

**Modos de cada modelo:**

- *cong* (congelado): o modelo nunca mais muda.
- *adapt* (adaptativo): o modelo continua aprendendo sem supervisão. No GWR, os nós novos recebem classe só pela propagação no grafo.
- As duas versões partem do mesmo modelo treinado.

**Variantes e comparação:**

- *+nov*: regra conservadora em que pontos com novidade acima do limiar de crescimento do próprio GWR (1 − a_T ≈ 0,65) viram "não pousável".
- k-NN treinado só nos 96 pontos rotulados, como referência.
- Capacidade igualada: `--protos 30` deixa o GWR com cerca de 100 nós após o treino, como o SOM 10×10 e o k-means com 100 grupos.

**Imagens de treino.** Para não depender de uma escolha, repeti o experimento com 4 imagens. A regra foi: em cada voo com as três classes, a imagem mais equilibrada. Ficaram t013 (voo_00), t078 (voo_01), t247 (voo_03) e t391 (voo_04); o voo_02 não tem água.

```bash
python experimento_uma_imagem.py --root sdd_prep --train-image t013_572
```

Resultado (média das 4 imagens de treino; 399 imagens avaliadas em cada):

| Modelo | mIoU | Pousável | Não pousável | Água | Falso pousável ↓ | Precisão pousável ↑ |
|---|---|---|---|---|---|---|
| **GWR-cong+nov** | **0,312 ± 0,025** | 0,533 | **0,334** | 0,067 | **0,277** | **0,836** |
| k-NN (só os 96 pontos) | 0,290 ± 0,039 | 0,576 | 0,237 | 0,058 | 0,439 | 0,789 |
| GWR-cong | 0,279 ± 0,038 | 0,538 | 0,238 | 0,061 | 0,392 | 0,795 |
| k-means-cong | 0,270 ± 0,046 | 0,520 | 0,241 | 0,049 | 0,403 | 0,773 |
| SOM-cong | 0,268 ± 0,024 | 0,522 | 0,240 | 0,042 | 0,390 | 0,784 |
| k-means-adapt | 0,226 ± 0,044 | 0,416 | 0,216 | 0,045 | 0,387 | 0,736 |
| SOM-adapt | 0,197 ± 0,039 | 0,374 | 0,195 | 0,021 | 0,392 | 0,716 |
| GWR-adapt | 0,188 ± 0,044 | 0,376 | 0,163 | 0,025 | 0,405 | 0,736 |

"Falso pousável" é a fração dos pixels não pousáveis ou de água classificados como pousáveis, o erro perigoso.

Conclusões:

1. **Uma imagem não cobre a variedade de aparências do SDD.** Todos os métodos ficam entre 0,19 e 0,31 de mIoU, e a água é praticamente irrecuperável. O modelo aprende a aparência da água *daquela* imagem: uma piscina clara em t013, um lago escuro em t247. Depois, generaliza para qualquer região com aparência parecida (cascalho claro, sombras, vegetação escura).
2. **A adaptação não supervisionada piora todos os modelos.** Sem nenhuma fonte de rótulo, os protótipos se deslocam e os novos nós herdam classes por vizinhança, o que gera deriva semântica. Para adaptar é preciso algum sinal de supervisão, ainda que fraco (por exemplo, geometria por profundidade ou LiDAR).
3. **A vantagem do GWR nesse cenário é a novidade, não a adaptação.** A regra "o que o mapa não reconhece não é pousável" foi a melhor ou empatou nas 4 imagens de treino. Ela reduz o falso pousável de ~0,39–0,44 para 0,28 e eleva a precisão do pousável para 0,84. Para pouso de emergência, esse é o resultado mais relevante. A regra só funciona no modo congelado: com adaptação, os nós novos absorvem a novidade.

#### Cenário binário: pousável × não pousável (`--binario`)

Neste cenário, a água é fundida em "não pousável". As 4 imagens de treino são as mesmas, e os 96 pontos rotulados também.

```bash
python experimento_uma_imagem.py --root sdd_prep --train-image t013_572 --binario
```

| Modelo (média das 4) | mIoU | IoU pousável | IoU não pousável | Falso pousável ↓ | Precisão pousável ↑ | 3 classes fundidas depois: mIoU / falso pousável |
|---|---|---|---|---|---|---|
| **GWR-cong+nov** | **0,451 ± 0,021** | 0,514 | **0,388** | **0,245** | **0,847** | 0,459 / 0,277 |
| k-NN | 0,447 ± 0,027 | 0,554 | 0,340 | 0,386 | 0,802 | 0,450 / 0,439 |
| GWR-cong | 0,430 ± 0,018 | 0,519 | 0,340 | 0,350 | 0,807 | 0,434 / 0,392 |
| SOM-cong | 0,425 ± 0,025 | 0,522 | 0,327 | 0,390 | 0,784 | 0,425 / 0,390 |
| k-means-cong | 0,422 ± 0,047 | 0,519 | 0,325 | 0,399 | 0,775 | 0,421 / 0,403 |
| k-means-adapt | 0,353 ± 0,061 | 0,415 | 0,291 | 0,383 | 0,737 | 0,353 / 0,387 |
| SOM-adapt | 0,322 ± 0,058 | 0,374 | 0,270 | 0,392 | 0,716 | 0,322 / 0,392 |
| GWR-adapt | 0,298 ± 0,060 | 0,318 | 0,279 | 0,318 | 0,756 | 0,318 / 0,405 |

A última coluna vem dos experimentos de 3 classes (seção anterior), com água e não pousável fundidos só na avaliação.

Conclusões:

1. **No mIoU, GWR-cong+nov e k-NN empatam** (0,451 contra 0,447): cada um vence em 2 das 4 imagens de treino.
2. **Na segurança, o GWR-cong+nov é claramente melhor.** O falso pousável cai de 0,386 (k-NN) e ~0,39–0,40 (SOM, k-means) para 0,245, e a precisão do pousável sobe para 0,847. O custo no IoU pousável é pequeno em relação ao GWR sem a regra (0,514 contra 0,519).
3. **Treinar com 3 classes e fundir depois dá praticamente o mesmo resultado** que treinar binário (0,459 contra 0,451). Rotular a água como classe separada não ajuda nem atrapalha; no binário, o falso pousável do GWR+nov é um pouco menor.
4. **A adaptação sem rótulos continua piorando todos os modelos**, como no cenário de 3 classes.

`--dataset` define os nomes e a paleta das classes:

| Valor | Classes |
|---|---|
| `ruralscapes` | 12 classes |
| `uavid` | Paleta oficial do UAVid, 8 classes |
| `uavid4` | Variante de 4 classes (estático / via / dinâmico / *clutter*) usada em `maliksyria/Semantic_Segmentation_UAVid` |
| `sdd` | Semantic Drone Dataset, 24 classes em índices; 0 e 23 são ignoradas |

Máscaras coloridas são convertidas por igualdade exata de cor, depois de redimensionadas. Cores fora da paleta são ignoradas na avaliação.

### Opções úteis

| Opção | Efeito |
|---|---|
| `--label-frac` | Fração das amostras de treino de cada frame que recebe rótulo. O padrão é 1%. |
| `--label-frames N` | Rótulos apenas nos N primeiros frames; depois disso, adaptação sem supervisão. |
| `--videos A B` | Ordem dos vídeos. A fronteira entre eles funciona como mudança de domínio. |
| `--protos` | Tamanho-alvo da calibração do GWR. Valores menores geram uma rede mais compacta. |
| `--max-nodes` | Capacidade máxima de nós. |

### Saídas

- `metrics.csv`: uma linha por frame e por método.
- `summary.json`: mIoU agregado, IoU por classe, nós e tempos.
- `curvas.png`: mIoU, número de nós e novidade ao longo do voo.
- `qual_*.png`: exemplos qualitativos.

## Notas

- O protocolo é **prequencial**: cada frame é avaliado *antes* de ser usado no treino.
- O SOM de referência usa o `train_SOM` original, com taxa de aprendizado e raio que decaem ao longo dos frames. Sem esse decaimento, cada chamada reiniciaria os valores e o mapa nunca se refinaria.
- As cores das máscaras PNG vêm do `CLASSES_BGR` do código oficial `ruralscapes-nn`. Se as máscaras do seu download usarem outra paleta, ajuste `RURAL_COLORS` ou use os rótulos `.npz`.
