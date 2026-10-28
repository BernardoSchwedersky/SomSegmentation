# -*- coding: utf-8 -*-
"""
Experimento: GWR semi-supervisionado para segmentação online de vídeo aéreo
(Ruralscapes), comparado com o SOM de somSeg.py e com k-means online.

Protocolo prequencial ("testa e depois treina"): os frames rotulados de cada
vídeo são processados em ordem temporal. Em cada frame, todos os métodos
primeiro segmentam o frame (avaliação), depois treinam com uma amostra de
patches dele, em que apenas uma fração `--label-frac` dos patches tem rótulo.

Uso:
  # Ruralscapes (formato oficial: rgb_frames/<video>_all/*.jpg e
  # labels/<video>/seg_*.npz, ou máscaras PNG coloridas)
  python ruralscapes_gwr.py --root /caminho/ruralscapes --videos DJI_0043 DJI_0051

  # UAVid (uavid_*/seqN/Images/*.png + uavid_*/seqN/Labels/*.png)
  python ruralscapes_gwr.py --dataset uavid --root /caminho/uavid --videos seq1 seq2

  # Teste de sanidade: voo simulado sobre uma única imagem rotulada
  python ruralscapes_gwr.py --dataset uavid4 --pan img.png mask.png

  # Validação com sequência sintética com mudança de "estação" no meio
  python ruralscapes_gwr.py --synthetic
"""

import argparse
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwrSeg  # noqa: E402
import somSeg  # noqa: E402

# Definição das classes de cada dataset: nomes, cores RGB das máscaras e
# agrupamento orientado a pouso de emergência (proposta; ajuste conforme o
# artigo). Rótulos .npz do Ruralscapes seguem a ordem dos nomes.
DATASETS = {
    # Cores convertidas do CLASSES_BGR do código oficial ruralscapes-nn
    'ruralscapes': dict(
        classes=['land', 'forest', 'residential', 'haystack', 'road', 'church',
                 'car', 'water', 'sky', 'hill', 'person', 'fence'],
        colors=[(0, 255, 0), (0, 127, 0), (255, 255, 0), (255, 127, 0),
                (255, 255, 255), (255, 0, 255), (127, 127, 127), (0, 0, 255),
                (0, 255, 255), (127, 127, 63), (255, 0, 0), (127, 127, 0)],
        landing=[0, 1, 1, 1, 0, 1, 1, 2, 3, 1, 1, 1]),
    # Paleta oficial do UAVid (8 classes)
    'uavid': dict(
        classes=['clutter', 'building', 'road', 'tree', 'low_vegetation',
                 'moving_car', 'static_car', 'human'],
        colors=[(0, 0, 0), (128, 0, 0), (128, 64, 128), (0, 128, 0),
                (128, 128, 0), (64, 0, 128), (192, 0, 192), (64, 64, 0)],
        landing=[1, 1, 0, 1, 0, 1, 1, 1]),
    # Variante de 4 classes usada em maliksyria/Semantic_Segmentation_UAVid
    'uavid4': dict(
        classes=['static', 'road', 'dynamic', 'clutter'],
        colors=[(128, 0, 0), (0, 128, 0), (0, 0, 128), (0, 0, 0)],
        landing=[1, 0, 1, 1]),
    # Semantic Drone Dataset (TU Graz): máscaras em índices 0-23 (tons de
    # cinza); 0 = unlabeled e 23 = conflicting são ignorados na avaliação.
    # Cores do class_dict do dataset, usadas só para visualização.
    'sdd': dict(
        classes=['unlabeled', 'paved-area', 'dirt', 'grass', 'gravel', 'water',
                 'rocks', 'pool', 'vegetation', 'roof', 'wall', 'window', 'door',
                 'fence', 'fence-pole', 'person', 'dog', 'car', 'bicycle', 'tree',
                 'bald-tree', 'ar-marker', 'obstacle', 'conflicting'],
        colors=[(0, 0, 0), (128, 64, 128), (130, 76, 0), (0, 102, 0),
                (112, 103, 87), (28, 42, 168), (48, 41, 30), (0, 50, 89),
                (107, 142, 35), (70, 70, 70), (102, 102, 156), (254, 228, 12),
                (254, 148, 12), (190, 153, 153), (153, 153, 153), (255, 22, 96),
                (102, 51, 0), (9, 143, 150), (119, 11, 32), (51, 51, 0),
                (190, 250, 190), (112, 150, 146), (2, 135, 115), (255, 0, 0)],
        # pousável: pavimento, terra, grama, cascalho, marcador AR (solo plano)
        landing=[1, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 0, 1, 1],
        ignore=[0, 23]),
}

LANDING_CLASSES = ['pousavel', 'nao_pousavel', 'agua', 'ceu']
LANDING_COLORS = np.array([(80, 200, 80), (200, 60, 60), (40, 90, 220),
                           (150, 220, 250)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


_LABEL_DIR_NAMES = ('labels', 'label', 'gt', 'masks', 'mask')


def find_pairs(root, videos=None):
    """Encontra pares (frame, rótulo) agrupados por vídeo/sequência.

    Formatos aceitos:
      Ruralscapes: rgb_frames/<video>_all/<frame>.jpg + labels/<video>/seg_<frame>.npz
      UAVid:       uavid_*/seqN/Images/<frame>.png  + uavid_*/seqN/Labels/<frame>.png

    Rótulos são .npz/.npy/.png em pastas cujo caminho contém 'label' (ou
    gt/mask). O frame correspondente é procurado primeiro na pasta irmã
    (Images ao lado de Labels) e depois pelo nome em qualquer pasta, sem o
    prefixo 'seg_'.
    """
    imgs, labs = {}, []
    for dirpath, _, files in os.walk(root):
        low = dirpath.lower()
        is_label_dir = ('label' in low or os.path.basename(low) in _LABEL_DIR_NAMES)
        for f in files:
            stem, ext = os.path.splitext(f)
            ext = ext.lower()
            p = os.path.join(dirpath, f)
            if is_label_dir and ext in ('.npz', '.png', '.npy'):
                labs.append(p)
            elif not is_label_dir and ext in ('.jpg', '.jpeg', '.png'):
                imgs.setdefault(stem, p)
    seqs = {}
    for lp in labs:
        stem = os.path.splitext(os.path.basename(lp))[0]
        stem = stem[4:] if stem.startswith('seg_') else stem
        label_dir = os.path.dirname(lp)
        ip = None
        # Pasta irmã (ex.: seqN/Labels -> seqN/Images): nomes repetem entre sequências
        if os.path.basename(label_dir).lower() in _LABEL_DIR_NAMES:
            parent = os.path.dirname(label_dir)
            for sib in os.listdir(parent):
                cand_dir = os.path.join(parent, sib)
                if sib.lower() in _LABEL_DIR_NAMES or not os.path.isdir(cand_dir):
                    continue
                for e in ('.png', '.jpg', '.jpeg'):
                    c = os.path.join(cand_dir, stem + e)
                    if os.path.exists(c):
                        ip = c
                        break
                if ip:
                    break
            video = os.path.basename(parent)
        else:
            video = os.path.basename(label_dir)
        if ip is None:
            ip = imgs.get(stem)
        if ip is None:
            continue
        if videos and not any(v in video for v in videos):
            continue
        seqs.setdefault(video, []).append((ip, lp, stem))
    for v in seqs:
        seqs[v].sort(key=lambda t: _natural_key(t[2]))
    order = sorted(seqs, key=_natural_key)
    if videos:  # respeita a ordem pedida na linha de comando
        order = sorted(order, key=lambda v: min(i for i, q in enumerate(videos) if q in v))
    return [(v, seqs[v]) for v in order]


def rgb_to_index(rgb, colors):
    """Converte máscara colorida (H, W, 3) em índices por igualdade exata de
    cor; cores fora da paleta viram -1 (ignoradas)."""
    rgb = np.asarray(rgb).astype(np.int64)
    code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    pal = np.asarray(colors, dtype=np.int64)
    pcode = (pal[:, 0] << 16) | (pal[:, 1] << 8) | pal[:, 2]
    lab = np.full(code.shape, -1, dtype=int)
    for i, c in enumerate(pcode):
        lab[code == c] = i
    return lab


def load_label(path, shape_hw, colors):
    """Carrega rótulo como mapa de índices (H, W) já no tamanho de trabalho,
    -1 = ignorar. Máscaras coloridas são redimensionadas (vizinho mais
    próximo) antes da conversão, para não estourar memória em frames 4K."""
    ext = os.path.splitext(path)[1].lower()
    size = (shape_hw[1], shape_hw[0])
    if ext in ('.npz', '.npy'):
        arr = np.load(path)['arr_0'] if ext == '.npz' else np.load(path)
        if arr.ndim == 3:  # one-hot (H, W, K)
            lab = np.argmax(arr, axis=-1)
            lab[arr.max(-1) <= 0] = -1
        else:
            lab = arr.astype(int)
        im = Image.fromarray((lab + 1).astype(np.uint8)).resize(size, Image.NEAREST)
        return np.asarray(im).astype(int) - 1
    im = Image.open(path)
    if im.mode in ('L', 'I', 'I;16'):  # máscara já em índices
        return np.asarray(im.resize(size, Image.NEAREST)).astype(int)
    rgb = np.asarray(im.convert('RGB').resize(size, Image.NEAREST))
    return rgb_to_index(rgb, colors)


def load_frame(img_path, lab_path, width, colors):
    im = Image.open(img_path).convert('RGB')
    h = int(round(im.height * width / im.width))
    im = np.asarray(im.resize((width, h), Image.BILINEAR))
    return im, load_label(lab_path, (h, width), colors)


def make_pan_sequence(img_path, lab_path, colors, n_frames=40, crop=(1024, 540),
                      width=480, seed=0):
    """Simula um voo sobre uma única imagem grande: uma janela `crop`
    percorre a imagem em zigue-zague e cada recorte vira um frame.
    Serve apenas como teste de sanidade quando não há sequências rotuladas."""
    big = np.asarray(Image.open(img_path).convert('RGB'))
    H, W = big.shape[:2]
    cw, ch = min(crop[0], W), min(crop[1], H)
    rows = max(1, int(np.ceil(n_frames / 10)))
    lab_full = load_label(lab_path, (H, W), colors)
    frames = []
    for t in range(n_frames):
        r = t // 10
        c = t % 10 if r % 2 == 0 else 9 - t % 10
        y0 = int((H - ch) * (r / max(rows - 1, 1)))
        x0 = int((W - cw) * (c / 9))
        h = int(round(ch * width / cw))
        img = np.asarray(Image.fromarray(big[y0:y0 + ch, x0:x0 + cw]).resize((width, h), Image.BILINEAR))
        lab = Image.fromarray((lab_full[y0:y0 + ch, x0:x0 + cw] + 1).astype(np.uint8))
        lab = np.asarray(lab.resize((width, h), Image.NEAREST)).astype(int) - 1
        frames.append((img, lab))
    return frames


def make_synthetic_sequence(n_frames=60, H=180, W=320, n_classes=5, shift_at=30, seed=0):
    """Sequência sintética de 'voo': janela deslizando sobre um mosaico de
    regiões texturizadas (Voronoi). A partir de `shift_at` a aparência das
    classes muda (troca de estação), mantendo a geometria das classes."""
    rs = np.random.RandomState(seed)
    MH, MW = H * 2, W * 4
    seeds = rs.rand(60, 2) * [MH, MW]
    cls_of_seed = rs.randint(0, n_classes, len(seeds))
    yy, xx = np.mgrid[0:MH, 0:MW]
    d = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    labels = cls_of_seed[np.argmin(d, -1)]

    from scipy.ndimage import gaussian_filter

    def render(base, tex_sigma, tex_amp):
        img = np.zeros((MH, MW, 3))
        for c in range(n_classes):
            noise = gaussian_filter(rs.randn(MH, MW), tex_sigma[c])
            noise /= noise.std() + 1e-9
            m = labels == c
            img[m] = base[c] + tex_amp[c] * noise[m, None] * np.array([1, 0.9, 0.8])
        return np.clip(img, 0, 255).astype(np.uint8)

    base1 = rs.uniform(40, 220, (n_classes, 3))
    sig = rs.uniform(0.6, 4.0, n_classes)
    amp = rs.uniform(8, 35, n_classes)
    mos1 = render(base1, sig, amp)
    base2 = np.clip(base1 * rs.uniform(0.6, 1.4, (n_classes, 3)) + rs.uniform(-40, 40, (n_classes, 3)), 0, 255)
    mos2 = render(base2, sig * rs.uniform(0.7, 1.5, n_classes), amp)

    frames = []
    for t in range(n_frames):
        x0 = int((MW - W) * t / max(n_frames - 1, 1))
        y0 = int((MH - H) * (0.5 + 0.5 * np.sin(t / 7.0)))
        mos = mos1 if t < shift_at else mos2
        frames.append((mos[y0:y0 + H, x0:x0 + W], labels[y0:y0 + H, x0:x0 + W]))
    return frames


# ---------------------------------------------------------------------------
# Métodos
# ---------------------------------------------------------------------------

class GWRMethod:
    def __init__(self, name, n_classes, args, propagate=True, conflict=True, semi=True):
        self.name, self.K, self.args = name, n_classes, args
        self.propagate, self.conflict, self.semi = propagate, conflict, semi
        self.net = None

    def _init(self, X):
        scale = gwrSeg.calibrate_dist_scale(X, n_protos=self.args.protos)
        self.net = gwrSeg.create_gwr(X.shape[1], n_classes=self.K,
                                     max_nodes=self.args.max_nodes,
                                     init_samples=X, dist_scale=scale)

    def train(self, X, y):
        if self.net is None:
            self._init(X)
        gwrSeg.train_GWR(self.net, X, y, epochs=1, semi=self.semi,
                         conflict=self.conflict, show_progress=False)
        if self.propagate:
            gwrSeg.label_propagation(self.net)

    def predict(self, X, shape=None):
        """Classifica a grade. Com --out-factor f > 1, o GWR só avalia 1 a
        cada f pontos em cada direção (grade f vezes mais grossa) e as
        probabilidades e a novidade são interpoladas (bilinear) de volta à
        grade de avaliação, reduzindo o custo de inferência em ~f^2."""
        if self.net is None:
            return np.zeros(len(X), int), None
        f = getattr(self.args, 'out_factor', 1)
        if f <= 1 or shape is None:
            o = gwrSeg.predict_GWR(self.net, X, use_propagation=self.propagate)
            return o['label'], o['novelty']
        import cv2
        Hs, Ws = shape
        Xg = X.reshape(Hs, Ws, -1)[::f, ::f]
        hc, wc = Xg.shape[:2]
        o = gwrSeg.predict_GWR(self.net, Xg.reshape(hc * wc, -1), use_propagation=self.propagate)
        # Interpolação alinhada aos pontos da grade grossa (cantos coincidem)
        P = o['prob'].reshape(hc, wc, -1).astype(np.float32)
        nov = o['novelty'].reshape(hc, wc).astype(np.float32)
        ys = np.minimum(np.arange(Hs) / f, hc - 1).astype(np.float32)
        xs = np.minimum(np.arange(Ws) / f, wc - 1).astype(np.float32)
        mx, my = np.meshgrid(xs, ys)
        Pu = np.stack([cv2.remap(np.ascontiguousarray(P[:, :, k]), mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE) for k in range(P.shape[2])], -1)
        nu = cv2.remap(nov, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Pu.reshape(Hs * Ws, -1).argmax(1), nu.reshape(-1)

    def n_nodes(self):
        return gwrSeg.n_nodes(self.net) if self.net is not None else 0


class CountLabeler:
    """Rótulo por nó/cluster por contagem (maioria) das amostras rotuladas."""
    def __init__(self, n_units, K):
        self.counts = np.zeros((n_units, K))

    def update(self, units, y):
        m = y >= 0
        np.add.at(self.counts, (units[m], y[m]), 1)

    def labels(self, units):
        tot = self.counts.sum(0)
        fallback = int(np.argmax(tot)) if tot.sum() > 0 else 0
        lab = np.argmax(self.counts, 1)
        lab[self.counts.sum(1) == 0] = fallback
        return lab[units]


class SOMMethod:
    """SOM de somSeg.py treinado online (uma passada por frame)."""
    def __init__(self, name, n_classes, args, lr0=0.1, lr_min=0.02,
                 r0=4.0, r_min=0.25, decay_frames=10.0):
        self.name, self.K, self.args = name, n_classes, args
        self.som = None
        # Decaimento da taxa e do raio ao longo dos frames (sem isso, cada
        # chamada a train_SOM reinicia lr e raio, e o mapa nunca se refina)
        self.lr0, self.lr_min, self.r0, self.r_min = lr0, lr_min, r0, r_min
        self.decay_frames, self.t = decay_frames, 0

    def _bmus(self, X):
        Wf = self.som['weights'].reshape(-1, X.shape[1])
        w2 = (Wf ** 2).sum(1)
        out = np.empty(len(X), int)
        for s in range(0, len(X), 8192):
            xb = X[s:s + 8192]
            out[s:s + 8192] = np.argmin(w2[None] - 2 * xb @ Wf.T, 1)
        return out

    def train(self, X, y):
        sx, sy = self.args.som_shape
        if self.som is None:
            self.som = somSeg.create_map(sx, sy, X.shape[1])
            idx = gwrSeg.rand.choice(len(X), sx * sy, replace=len(X) < sx * sy)
            self.som['weights'] = X[idx].reshape(sx, sy, X.shape[1]).astype(float)
            self.lab = CountLabeler(sx * sy, self.K)
        f = np.exp(-self.t / self.decay_frames)
        lr = max(self.lr_min, self.lr0 * f)
        r2 = max(self.r_min, self.r0 * f)
        self.som = somSeg.train_SOM(self.som, X.copy(), learn_rate=lr,
                                    radius_sq=r2, epochs=1)
        self.t += 1
        self.lab.update(self._bmus(X), y)

    def predict(self, X):
        if self.som is None:
            return np.zeros(len(X), int), None
        return self.lab.labels(self._bmus(X)), None

    def n_nodes(self):
        return int(np.prod(self.args.som_shape))


class KMeansMethod:
    def __init__(self, name, n_classes, args):
        from sklearn.cluster import MiniBatchKMeans
        self.name, self.K = name, n_classes
        self.k = int(np.prod(args.som_shape))
        self.km = MiniBatchKMeans(n_clusters=self.k, random_state=0, n_init=3,
                                  batch_size=1024)
        self.fitted = False
        self.lab = CountLabeler(self.k, n_classes)

    def train(self, X, y):
        self.km.partial_fit(X)
        self.fitted = True
        self.lab.update(self.km.predict(X), y)

    def predict(self, X):
        if not self.fitted:
            return np.zeros(len(X), int), None
        return self.lab.labels(self.km.predict(X)), None

    def n_nodes(self):
        return self.k


# ---------------------------------------------------------------------------
# Métricas e figuras
# ---------------------------------------------------------------------------

def confusion(gt, pred, K):
    m = gt >= 0
    return np.bincount(K * gt[m] + pred[m], minlength=K * K).reshape(K, K)


def iou_from_conf(C):
    tp = np.diag(C).astype(float)
    denom = C.sum(0) + C.sum(1) - tp
    present = C.sum(1) > 0  # classes presentes no GT
    iou = np.divide(tp, denom, out=np.full(len(tp), np.nan), where=denom > 0)
    iou[~present] = np.nan
    return iou


def colorize(lab, colors):
    out = np.zeros(lab.shape + (3,), np.uint8)
    m = lab >= 0
    out[m] = colors[lab[m]]
    return out


def save_curves(rows, methods, boundaries, path, title):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for name in methods:
        r = [x for x in rows if x['method'] == name]
        t = [x['step'] for x in r]
        miou = np.array([x['miou'] for x in r])
        # Média móvel que ignora NaN e não puxa as bordas para zero
        k = max(1, len(miou) // 30)
        valid = ~np.isnan(miou)
        num = np.convolve(np.where(valid, miou, 0.0), np.ones(k), mode='same')
        den = np.convolve(valid.astype(float), np.ones(k), mode='same')
        sm = np.divide(num, den, out=np.full(len(miou), np.nan), where=den > 0)
        ax[0].plot(t, sm, label=name)
        ax[1].plot(t, [x['n_nodes'] for x in r], label=name)
        nov = [x['novelty'] for x in r]
        if not all(v is None for v in nov):
            ax[2].plot(t, [np.nan if v is None else v for v in nov], label=name)
    for b, lbl in boundaries:
        for a in ax:
            a.axvline(b, color='k', ls='--', lw=0.8)
        ax[0].text(b, 1.0, ' ' + lbl, fontsize=8, va='top')
    ax[0].set_ylabel('mIoU (prequencial)')
    ax[0].set_ylim(0, 1)
    ax[1].set_ylabel('nós')
    ax[2].set_ylabel('novidade média (GWR)')
    ax[2].set_xlabel('frame')
    ax[0].legend(fontsize=8, ncol=3)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_qualitative(img, gt, preds, novelty, colors, path, title):
    n = 3 + len(preds)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 2.6))
    ax[0].imshow(img); ax[0].set_title('RGB')
    ax[1].imshow(colorize(gt, colors)); ax[1].set_title('GT')
    for i, (name, p) in enumerate(preds.items()):
        ax[2 + i].imshow(colorize(p, colors)); ax[2 + i].set_title(name)
    im = ax[-1].imshow(novelty, cmap='magma', vmin=0, vmax=1)
    ax[-1].set_title('novidade GWR')
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_qualitative_grid(entries, colors, class_names, path):
    """Grade com uma linha por imagem: RGB, GT, um mapa por método e a
    novidade do GWR. `entries` é uma lista de dicionários com img, gt, preds
    (nome -> mapa), miou (nome -> valor), novelty e title."""
    from matplotlib.patches import Patch
    names = list(entries[0]['preds'])
    ncol = 3 + len(names)
    nrow = len(entries)
    h, w = entries[0]['gt'].shape
    cw = 3.0
    fig, ax = plt.subplots(nrow, ncol, figsize=(cw * ncol, cw * h / w * nrow + 1.2),
                           squeeze=False)
    for r, e in enumerate(entries):
        ax[r, 0].imshow(e['img'])
        ax[r, 1].imshow(colorize(e['gt'], colors))
        for i, n in enumerate(names):
            ax[r, 2 + i].imshow(colorize(e['preds'][n], colors))
            ax[r, 2 + i].set_xlabel('mIoU %.2f' % e['miou'][n], fontsize=8, labelpad=2)
        im = ax[r, -1].imshow(e['novelty'], cmap='magma', vmin=0, vmax=1)
        ax[r, -1].set_xlabel('média %.2f' % float(np.mean(e['novelty'])), fontsize=8, labelpad=2)
        ax[r, 0].set_ylabel(e['title'], fontsize=8)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    for c, t in enumerate(['RGB', 'GT (anotação)'] + names + ['novidade GWR']):
        ax[0, c].set_title(t, fontsize=10)
    present = sorted({int(v) for e in entries for v in np.unique(e['gt']) if v >= 0} |
                     {int(v) for e in entries for p in e['preds'].values() for v in np.unique(p)})
    handles = [Patch(color=np.array(colors[k]) / 255.0, label=class_names[k]) for k in present]
    handles.append(Patch(color='black', label='ignorado'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 0.97, 1))
    cax = fig.add_axes([0.975, 0.1, 0.008, 0.8])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.out, exist_ok=True)
    rs = np.random.RandomState(args.seed)
    gwrSeg.rand.seed(args.seed)

    # Monta a lista de frames: [(video, frame_id, loader)]
    if args.synthetic:
        seq = make_synthetic_sequence(n_frames=args.synthetic_frames,
                                      shift_at=args.synthetic_frames // 2,
                                      seed=args.seed)
        K = 5
        class_names = ['c%d' % i for i in range(K)]
        colors = np.array([(230, 200, 60), (40, 140, 40), (90, 90, 200),
                           (200, 80, 80), (200, 200, 200)], np.uint8)
        items = [('estacao_A' if t < len(seq) // 2 else 'estacao_B', 't%03d' % t,
                  (lambda f=f: f)) for t, f in enumerate(seq)]
    else:
        ds = DATASETS[args.dataset]
        mask_colors = ds['colors']
        landing_map = np.array(ds['landing'])
        ignore = ds.get('ignore', [])
        if args.classes == 'landing':
            K, class_names, colors = 4, LANDING_CLASSES, LANDING_COLORS
        else:
            K, class_names = len(ds['classes']), ds['classes']
            colors = np.array(mask_colors, np.uint8)
        items = []
        if args.pan:
            seq = make_pan_sequence(args.pan[0], args.pan[1], mask_colors,
                                    n_frames=args.pan_frames, width=args.width,
                                    seed=args.seed)
            name = os.path.splitext(os.path.basename(args.pan[0]))[0]
            items = [(name, 'pan%03d' % t, (lambda f=f: f)) for t, f in enumerate(seq)]
            print('voo simulado sobre %s: %d recortes' % (args.pan[0], len(items)))
        else:
            pairs = find_pairs(args.root, args.videos)
            if not pairs:
                sys.exit('Nenhum par frame/rótulo encontrado em %s' % args.root)
            for v, lst in pairs:
                lst = lst[::args.frame_step][:args.max_frames]
                print('sequência %s: %d frames rotulados' % (v, len(lst)))
                for ip, lp, stem in lst:
                    items.append((v, stem, (lambda ip=ip, lp=lp:
                                            load_frame(ip, lp, args.width, mask_colors))))

    methods = [
        GWRMethod('GWR-semi', K, args),
        GWRMethod('GWR-sem-prop', K, args, propagate=False),
        GWRMethod('GWR-sem-conflito', K, args, conflict=False),
        SOMMethod('SOM', K, args),
        KMeansMethod('k-means', K, args),
    ]
    if args.methods:
        methods = [m for m in methods if m.name in args.methods]
    conf = {m.name: np.zeros((K, K), np.int64) for m in methods}
    conf_v = {m.name: {} for m in methods}  # confusão acumulada por sequência
    rows, boundaries = [], []
    prev_video = None
    qual_every = max(1, len(items) // args.n_qualitative)
    # Imagens da grade qualitativa: passos explícitos e/ou o frame central
    # de cada sequência
    grid_steps = set(args.qual_steps or [])
    if args.qual_per_video:
        vids = [it[0] for it in items]
        for v in dict.fromkeys(vids):
            idx = [i for i, x in enumerate(vids) if x == v]
            grid_steps.add(idx[len(idx) // 2])
    grid = []
    use_norm = args.normalize if args.normalize is not None else (args.features in ('filters', 'pyramid'))
    normalizer = gwrSeg.FeatureNormalizer(args.pca) if (use_norm or args.pca > 0) else None

    for step, (video, fid, loader) in enumerate(items):
        img, lab = loader()
        if not args.synthetic:
            if ignore:
                lab = np.where(np.isin(lab, ignore), -1, lab)
            if args.classes == 'landing':
                lab = np.where(lab >= 0, landing_map[np.clip(lab, 0, None)], -1)
        if video != prev_video:
            boundaries.append((step, video))
            prev_video = video
        t_feat = time.time()
        X, shape, (ys, xs) = gwrSeg.extract_features(img, args.patch, args.stride, args.features,
                                                     levels=args.levels)
        if normalizer is not None:
            if not normalizer.fitted:  # ajustado só com o 1º frame, sem rótulos
                normalizer.fit(X)
                if normalizer.P is not None:
                    print('PCA: %d componentes, %.1f%% da variância' %
                          (normalizer.n_pca, 100 * normalizer.explained))
            X = normalizer.transform(X)
        t_feat = (time.time() - t_feat) * 1000
        y = lab[ys][:, xs].reshape(-1)

        # Amostra de treino deste frame e rótulos esparsos
        tr = rs.choice(len(X), min(args.samples, len(X)), replace=False)
        y_tr = y[tr].copy()
        labeled_frame = (step % args.label_every == 0) and (args.label_frames < 0 or step < args.label_frames)
        mask = rs.rand(len(tr)) < args.label_frac
        y_tr[~(mask & labeled_frame)] = -1

        preds, nov_map = {}, None
        for m in methods:
            t0 = time.time()
            p, nov = m.predict(X, shape) if isinstance(m, GWRMethod) else m.predict(X)
            t_inf = (time.time() - t0) * 1000
            if step > 0:  # avaliação prequencial (o primeiro frame não tem modelo)
                C = confusion(y, p, K)
                conf[m.name] += C
                cv = conf_v[m.name].setdefault(video, np.zeros((K, K), np.int64))
                cv += C
                miou = float(np.nanmean(iou_from_conf(C))) if (y >= 0).any() else np.nan
            else:
                miou = np.nan
            t0 = time.time()
            m.train(X[tr], y_tr)
            t_tr = time.time() - t0
            row = dict(step=step, video=video, frame=fid, method=m.name, miou=miou,
                       acc=float((p[y >= 0] == y[y >= 0]).mean()) if step > 0 else np.nan,
                       n_nodes=m.n_nodes(), infer_ms=t_inf, train_s=t_tr,
                       novelty=float(nov.mean()) if nov is not None else None,
                       n_labeled=int((y_tr >= 0).sum()), feat_ms=t_feat)
            rows.append(row)
            preds[m.name] = p.reshape(shape)
            if m.name == 'GWR-semi' and nov is not None:
                nov_map = nov.reshape(shape)

        line = ' '.join('%s=%.3f' % (r['method'], r['miou']) for r in rows[-len(methods):])
        print('[%3d/%d] %s %s  %s' % (step + 1, len(items), video, fid, line), flush=True)

        if step > 0 and (step % qual_every == 0 or step == len(items) - 1) and nov_map is not None:
            sel = {k: v for k, v in preds.items() if k in ('GWR-semi', 'SOM', 'k-means')}
            save_qualitative(img[ys][:, xs], lab[ys][:, xs], sel, nov_map, colors,
                             os.path.join(args.out, 'qual_%03d.png' % step),
                             '%s %s' % (video, fid))
        if step in grid_steps and step > 0 and nov_map is not None:
            sel = {k: v for k, v in preds.items() if k in ('GWR-semi', 'SOM', 'k-means')}
            mi = {r['method']: r['miou'] for r in rows[-len(methods):]}
            grid.append(dict(img=img[ys][:, xs], gt=lab[ys][:, xs], preds=sel,
                             miou={k: mi[k] for k in sel}, novelty=nov_map,
                             title='%s\n%s (#%d)' % (video, fid, step)))

    if grid:
        save_qualitative_grid(grid, colors, class_names,
                              os.path.join(args.out, 'qual_grade.png'))

    # Resultados agregados
    summary = {'args': {k: v for k, v in vars(args).items()}, 'classes': class_names, 'methods': {}}
    print('\nResultado agregado (confusão acumulada, prequencial):')
    print('%-18s %8s %8s %7s %9s' % ('método', 'mIoU', 'pix-acc', 'nós', 'inf(ms)'))
    for m in methods:
        C = conf[m.name]
        iou = iou_from_conf(C)
        r = [x for x in rows if x['method'] == m.name]
        summary['methods'][m.name] = {
            'mIoU': float(np.nanmean(iou)),
            'pixel_acc': float(np.diag(C).sum() / max(C.sum(), 1)),
            'mIoU_per_video': {v: float(np.nanmean(iou_from_conf(Cv)))
                               for v, Cv in conf_v[m.name].items()},
            'conf_per_video': {v: Cv.tolist() for v, Cv in conf_v[m.name].items()},
            'IoU_per_class': {c: (None if np.isnan(v) else float(v)) for c, v in zip(class_names, iou)},
            'final_nodes': r[-1]['n_nodes'],
            'mean_infer_ms': float(np.mean([x['infer_ms'] for x in r])),
            'mean_feat_ms': float(np.mean([x['feat_ms'] for x in r])),
            'mean_train_s': float(np.mean([x['train_s'] for x in r])),
        }
        if isinstance(m, GWRMethod) and m.net is not None:
            summary['methods'][m.name]['growth'] = {k: int(v) for k, v in m.net['stats'].items()}
        s = summary['methods'][m.name]
        print('%-18s %8.3f %8.3f %7d %9.1f' % (m.name, s['mIoU'], s['pixel_acc'],
                                                s['final_nodes'], s['mean_infer_ms']))

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    import csv
    with open(os.path.join(args.out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_curves(rows, [m.name for m in methods], boundaries,
                os.path.join(args.out, 'curvas.png'),
                'Segmentação online prequencial (%s)' % (
                    'sintético' if args.synthetic else
                    args.dataset + (' — voo simulado' if args.pan else '')))
    print('Resultados em', os.path.abspath(args.out))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', help='pasta raiz do dataset')
    p.add_argument('--dataset', choices=sorted(DATASETS), default='ruralscapes',
                   help='define nomes e paleta de cores das classes')
    p.add_argument('--videos', nargs='*', help='vídeos/sequências (substrings), na ordem de processamento')
    p.add_argument('--pan', nargs=2, metavar=('IMAGEM', 'MASCARA'),
                   help='teste de sanidade: simula um voo sobre uma única imagem grande')
    p.add_argument('--pan-frames', type=int, default=40)
    p.add_argument('--synthetic', action='store_true', help='usa a sequência sintética')
    p.add_argument('--synthetic-frames', type=int, default=60)
    p.add_argument('--classes', choices=['full', 'landing'], default='full')
    p.add_argument('--width', type=int, default=480, help='largura de trabalho dos frames')
    p.add_argument('--features', choices=['patch', 'stats', 'filters', 'pyramid'], default='stats',
                   help="'filters' = banco de filtros convolucionais (cor multiescala, "
                        "Gabor invariante à rotação, LoG, gradiente, desvio local)")
    p.add_argument('--normalize', dest='normalize', action='store_true', default=None,
                   help='z-score ajustado no 1º frame (padrão: ligado só para filters)')
    p.add_argument('--no-normalize', dest='normalize', action='store_false')
    p.add_argument('--pca', type=int, default=0, help='nº de componentes PCA (0 = sem PCA)')
    p.add_argument('--levels', type=int, default=4,
                   help="níveis do encoder em pirâmide (--features pyramid)")
    p.add_argument('--out-factor', type=int, default=1,
                   help='GWR classifica uma grade N vezes mais grossa e interpola as probabilidades')
    p.add_argument('--patch', type=int, default=2, help='meia largura do patch (b)')
    p.add_argument('--stride', type=int, default=4, help='passo da grade de amostras')
    p.add_argument('--samples', type=int, default=3000, help='amostras de treino por frame')
    p.add_argument('--label-frac', type=float, default=0.01, help='fração das amostras com rótulo')
    p.add_argument('--label-every', type=int, default=1, help='rótulos a cada N frames')
    p.add_argument('--label-frames', type=int, default=-1, help='rótulos só nos N primeiros frames (-1 = todos)')
    p.add_argument('--frame-step', type=int, default=1, help='usa 1 a cada N frames rotulados')
    p.add_argument('--max-frames', type=int, default=10 ** 9, help='máximo de frames por vídeo')
    p.add_argument('--protos', type=int, default=30, help='tamanho-alvo usado na calibração da escala do GWR')
    p.add_argument('--max-nodes', type=int, default=600)
    p.add_argument('--som-shape', type=int, nargs=2, default=[10, 10])
    p.add_argument('--methods', nargs='*', help='subconjunto de métodos')
    p.add_argument('--n-qualitative', type=int, default=6)
    p.add_argument('--qual-steps', type=int, nargs='*',
                   help='passos (índice do frame) para a grade qualitativa qual_grade.png')
    p.add_argument('--qual-per-video', action='store_true',
                   help='inclui na grade o frame central de cada sequência')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='resultados_gwr')
    a = p.parse_args(argv)
    if not a.synthetic and not a.root and not a.pan:
        p.error('informe --root, --pan ou --synthetic')
    return a


if __name__ == '__main__':
    run(parse_args())
