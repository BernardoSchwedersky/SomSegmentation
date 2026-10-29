# -*- coding: utf-8 -*-
"""
Features de filtros convolucionais pré-treinados para a entrada do SOM.

Usa os dois primeiros blocos da VGG16 treinada no ImageNet (pesos públicos do
Keras, baixados uma vez de storage.googleapis.com), implementados em NumPy, sem
precisar de TensorFlow ou PyTorch:

    block1_conv1 -> block1_conv2 (64 filtros, resolução cheia)
    max pooling 2x2 -> block2_conv1 -> block2_conv2 (128 filtros, meia resolução)

A entrada é a composição em cor verdadeira (B04, B03, B02), com o contraste
esticado pelos percentis 2-98% da área (a mesma escala nas duas datas) e o
pré-processamento da VGG (BGR e subtração da média do ImageNet). As features de
block2 são ampliadas de volta para a resolução cheia. Como são 192 canais por
data, elas são reduzidas por PCA (ajustada com pixels das duas datas) a poucas
componentes, que entram no SOM como canais extras de cada data.

Dependência extra: pip install h5py
"""

import os
import urllib.request

import numpy as np

URL_PESOS = ("https://storage.googleapis.com/tensorflow/keras-applications/vgg16/"
             "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
MEDIA_IMAGENET_BGR = np.array([103.939, 116.779, 123.68])
CAMADAS = ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2"]

_pesos = None


def carregar_pesos(arquivo=os.path.join(os.path.expanduser("~"), ".cache", "somseg", "vgg16_notop.h5")):
    global _pesos
    if _pesos is None:
        import h5py
        if not os.path.exists(arquivo):
            os.makedirs(os.path.dirname(arquivo), exist_ok=True)
            urllib.request.urlretrieve(URL_PESOS, arquivo + ".tmp")
            os.replace(arquivo + ".tmp", arquivo)
        with h5py.File(arquivo, "r") as f:
            _pesos = {c: (f[c][c + "_W_1:0"][()], f[c][c + "_b_1:0"][()]) for c in CAMADAS}
    return _pesos

# Convolução 3x3 com borda de zeros ("same", como no Keras) seguida de ReLU.
# kernel: (3, 3, canais_entrada, canais_saida)
def conv3x3_relu(x, kernel, vies, bloco_linhas=64):
    h, w, c = x.shape
    pad = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    janelas = np.lib.stride_tricks.sliding_window_view(pad, (3, 3), axis=(0, 1))  # (h, w, c, 3, 3)
    k = kernel.transpose(2, 0, 1, 3).reshape(c * 9, -1)  # Mesma ordem (c, 3, 3) das janelas
    saida = np.empty((h, w, kernel.shape[3]), dtype=np.float32)
    for i in range(0, h, bloco_linhas):
        bloco = janelas[i:i + bloco_linhas].reshape(-1, c * 9)
        saida[i:i + bloco_linhas] = np.maximum(bloco @ k + vies, 0).reshape(-1, w, kernel.shape[3])
    return saida

def max_pool2(x):
    h, w, c = x.shape
    x = x[:h // 2 * 2, :w // 2 * 2]
    return x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))

# Composição RGB 0-255 com contraste pelos percentis das duas datas (mesma escala)
def rgb_0_255(refl_antes, refl_depois, invalidos=None):
    rgbs = [r[:, :, [2, 1, 0]] for r in (refl_antes, refl_depois)]  # B04, B03, B02 = R, G, B
    validos = np.ones(rgbs[0].shape[:2], bool) if invalidos is None else ~invalidos
    lo, hi = np.percentile(np.concatenate([r[validos] for r in rgbs]), [2, 98])
    return [np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1) * 255 for r in rgbs]

# Features dos dois blocos da VGG16 (192 canais na resolução cheia)
def features_vgg(rgb):
    p = carregar_pesos()
    x = (rgb[:, :, ::-1] - MEDIA_IMAGENET_BGR).astype(np.float32)  # RGB -> BGR, menos a média
    b1 = conv3x3_relu(conv3x3_relu(x, *p["block1_conv1"]), *p["block1_conv2"])
    b2 = conv3x3_relu(conv3x3_relu(max_pool2(b1), *p["block2_conv1"]), *p["block2_conv2"])
    b2 = np.repeat(np.repeat(b2, 2, axis=0), 2, axis=1)
    h, w = rgb.shape[:2]
    if b2.shape[:2] != (h, w):  # Dimensão ímpar: completa a última linha/coluna
        b2 = np.pad(b2, ((0, h - b2.shape[0]), (0, w - b2.shape[1]), (0, 0)), mode="edge")
    return np.concatenate([b1, b2], axis=2)

# Features VGG das duas datas reduzidas por PCA a n_componentes cada (mesma base
# para as duas datas, ajustada com pixels sorteados das duas)
def features_pca(refl_antes, refl_depois, invalidos=None, n_componentes=4, amostras=20000, seed=0):
    rgb_antes, rgb_depois = rgb_0_255(refl_antes, refl_depois, invalidos)
    f_antes, f_depois = features_vgg(rgb_antes), features_vgg(rgb_depois)
    h, w, c = f_antes.shape
    todos = np.concatenate([f_antes.reshape(-1, c), f_depois.reshape(-1, c)])
    rng = np.random.RandomState(seed)
    amostra = todos[rng.choice(len(todos), min(amostras, len(todos)), replace=False)]
    media, desvio = amostra.mean(0), amostra.std(0) + 1e-6
    _, _, vt = np.linalg.svd((amostra - media) / desvio, full_matrices=False)
    base = vt[:n_componentes].T
    projetar = lambda f: (((f.reshape(-1, c) - media) / desvio) @ base).reshape(h, w, n_componentes)
    return projetar(f_antes), projetar(f_depois)
