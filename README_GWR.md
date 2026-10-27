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
```

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
