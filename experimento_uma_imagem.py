# -*- coding: utf-8 -*-
"""
Protocolo "uma imagem": todos os modelos são treinados com UMA única imagem,
na qual só 1% dos pontos da grade recebem rótulo (pousável / não pousável /
água). Depois disso nenhum modelo recebe anotação alguma: as demais imagens
são classificadas usando apenas a própria imagem, e os modelos podem (modo
adaptativo) ou não (modo congelado) se atualizar de forma não supervisionada.

Modelos:
  GWR-adapt / GWR-cong     : GWR semi-supervisionado. No modo adaptativo o
                             grafo continua aprendendo (crescimento por
                             novidade, deslocamento dos nós, reciclagem) e os
                             rótulos dos nós novos vêm só da propagação no
                             grafo a partir dos nós rotulados na imagem de treino
  SOM-adapt / SOM-cong     : SOM de somSeg.py; rótulo de cada unidade pelos
                             pontos rotulados (maioria), unidades sem rótulo
                             herdam da unidade rotulada mais próxima
  k-means-adapt / -cong    : idem, com MiniBatchKMeans (partial_fit)
  k-NN                     : classificador k-NN só com os pontos rotulados
                             (sem estrutura não supervisionada)

Avaliação: todas as outras imagens, em ordem temporal; mIoU por confusão
acumulada, por voo e por classe.

Uso:
  python experimento_uma_imagem.py --root sdd_prep --train-image t013_572
"""

import argparse
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwrSeg  # noqa: E402
import somSeg  # noqa: E402
import ruralscapes_gwr as R  # noqa: E402


POUSAVEL, NAO_POUSAVEL, AGUA = 0, 1, 2


def safety(C):
    """Métricas de segurança para pouso a partir da confusão C[gt, pred]:
    false_landable = fração dos pixels não pousáveis ou água previstos como
    pousáveis (o erro perigoso); landable_precision = fração dos pixels
    previstos como pousáveis que realmente são."""
    C = np.asarray(C, float)
    bad = C[1:, :].sum()
    return dict(false_landable=float(C[1:, POUSAVEL].sum() / max(bad, 1)),
                landable_precision=float(C[POUSAVEL, POUSAVEL] / max(C[:, POUSAVEL].sum(), 1)))


# ---------------------------------------------------------------------------
# Rotulagem de protótipos (SOM / k-means)
# ---------------------------------------------------------------------------

def label_units(W, units_lab, y_lab, K):
    """Rótulo por unidade: maioria dos pontos rotulados que caíram nela;
    unidades sem nenhum ponto rotulado herdam o rótulo da unidade rotulada
    mais próxima (mesma regra usada para nós não informados do GWR)."""
    counts = np.zeros((len(W), K))
    np.add.at(counts, (units_lab, y_lab), 1)
    has = counts.sum(1) > 0
    lab = np.argmax(counts, 1)
    if (~has).any() and has.any():
        Wl, Wu = W[has], W[~has]
        d2 = (Wu ** 2).sum(1)[:, None] + (Wl ** 2).sum(1)[None] - 2 * Wu @ Wl.T
        lab[~has] = lab[has][np.argmin(d2, 1)]
    return lab


def nearest(W, X, batch=8192):
    w2 = (W ** 2).sum(1)
    out = np.empty(len(X), int)
    for s in range(0, len(X), batch):
        xb = X[s:s + batch]
        out[s:s + batch] = np.argmin(w2[None] - 2 * xb @ W.T, 1)
    return out


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class GWRModel:
    def __init__(self, name, K, args, adapt):
        self.name, self.K, self.args, self.adapt_on = name, K, args, adapt
        self.nov_rule = False  # regra conservadora: novo demais -> não pousável

    def fit(self, X, y):
        a = self.args
        scale = gwrSeg.calibrate_dist_scale(X, n_protos=a.protos)
        self.net = gwrSeg.create_gwr(X.shape[1], n_classes=self.K, max_nodes=a.max_nodes,
                                     init_samples=X, dist_scale=scale)
        gwrSeg.train_GWR(self.net, X, y, epochs=a.train_epochs, show_progress=False)
        self._relabel()

    def _relabel(self):
        gwrSeg.label_propagation(self.net)
        gwrSeg.fill_uninformed(self.net)

    def predict(self, X):
        o = gwrSeg.predict_GWR(self.net, X, use_propagation=True)
        lab = o['label']
        if self.nov_rule:
            # Limiar = o próprio critério de crescimento do GWR: a entrada está
            # longe demais de tudo o que o mapa conhece (a < a_T)
            thr = 1.0 - self.net['cfg']['act_thr']
            lab = np.where(o['novelty'] > thr, NAO_POUSAVEL, lab)
        return lab, o['novelty']

    def adapt(self, X):
        if not self.adapt_on:
            return
        # Sem rótulos: só crescimento por novidade, deslocamento e reciclagem
        gwrSeg.train_GWR(self.net, X, None, epochs=1, show_progress=False)
        self._relabel()

    def size(self):
        return gwrSeg.n_nodes(self.net)


class SOMModel:
    def __init__(self, name, K, args, adapt):
        self.name, self.K, self.args, self.adapt_on = name, K, args, adapt

    def fit(self, X, y):
        sx, sy = self.args.som_shape
        self.som = somSeg.create_map(sx, sy, X.shape[1])
        idx = gwrSeg.rand.choice(len(X), sx * sy, replace=False)
        self.som['weights'] = X[idx].reshape(sx, sy, -1).astype(float)
        self.som = somSeg.train_SOM(self.som, X.copy(), learn_rate=0.1, radius_sq=4,
                                    lr_decay=0.5, radius_decay=0.5,
                                    epochs=self.args.train_epochs)
        W = self._W()
        m = y >= 0
        self.lab = label_units(W, nearest(W, X[m]), y[m], self.K)

    def _W(self):
        return self.som['weights'].reshape(-1, self.som['weights'].shape[-1])

    def predict(self, X):
        return self.lab[nearest(self._W(), X)], None

    def adapt(self, X):
        if self.adapt_on:  # ajuste fino com taxa e raio pequenos, rótulos fixos
            self.som = somSeg.train_SOM(self.som, X.copy(), learn_rate=0.02,
                                        radius_sq=0.25, epochs=1)

    def size(self):
        return int(np.prod(self.args.som_shape))


class KMeansModel:
    def __init__(self, name, K, args, adapt):
        from sklearn.cluster import MiniBatchKMeans
        self.name, self.K, self.adapt_on = name, K, adapt
        self.k = int(np.prod(args.som_shape))
        self.km = MiniBatchKMeans(n_clusters=self.k, random_state=args.seed, n_init=3,
                                  batch_size=1024)

    def fit(self, X, y):
        self.km.fit(X)
        m = y >= 0
        self.lab = label_units(self.km.cluster_centers_, self.km.predict(X[m]), y[m], self.K)

    def predict(self, X):
        return self.lab[self.km.predict(X)], None

    def adapt(self, X):
        if self.adapt_on:
            self.km.partial_fit(X)

    def size(self):
        return self.k


class KNNModel:
    def __init__(self, name, K, args, adapt=False):
        from sklearn.neighbors import KNeighborsClassifier
        self.name, self.K = name, K
        self.clf = KNeighborsClassifier(n_neighbors=args.knn_k)

    def fit(self, X, y):
        m = y >= 0
        self.n = int(m.sum())
        self.clf.fit(X[m], y[m])

    def predict(self, X):
        return self.clf.predict(X), None

    def adapt(self, X):
        pass

    def size(self):
        return self.n


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def save_train_figure(img, gt, lab_pts, colors, class_names, path, title):
    """Imagem de treino, GT e os pontos rotulados que os modelos receberam."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.0 * gt.shape[0] / gt.shape[1] + 0.9))
    ax[0].imshow(img); ax[0].set_title('imagem de treino')
    ax[1].imshow(R.colorize(gt, colors)); ax[1].set_title('GT (não fornecido aos modelos)')
    ax[2].imshow(img, alpha=0.45)
    yy, xx, cc = lab_pts
    ax[2].scatter(xx, yy, c=np.array(colors)[cc] / 255.0, s=22, edgecolors='k', linewidths=0.6)
    ax[2].set_title('pontos rotulados fornecidos (%d = 1%%)' % len(cc))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    hd = [Patch(color=np.array(colors[k]) / 255.0, label=class_names[k]) for k in range(3)]
    fig.legend(handles=hd, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_curves(rows, names, boundaries, path, title):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    styles = {}
    for name in names:
        r = [x for x in rows if x['method'] == name]
        t = np.array([x['step'] for x in r])
        m = np.array([x['miou'] for x in r], float)
        k = max(1, min(15, len(m)))
        num = np.convolve(np.nan_to_num(m), np.ones(k), 'same')
        den = np.convolve((~np.isnan(m)).astype(float), np.ones(k), 'same')
        sm = np.divide(num, den, out=np.full(len(m), np.nan), where=den > 0)
        ls = '--' if name.endswith('cong') else '-'
        ln, = ax[0].plot(t, sm, ls, label=name, lw=1.3)
        styles[name] = ln.get_color()
        if name.startswith('GWR'):
            ax[1].plot(t, [x['size'] for x in r], ls, color=ln.get_color(), label=name)
    for b, lbl in boundaries:
        for a in ax:
            a.axvline(b, color='k', ls=':', lw=0.8)
        ax[0].text(b, 0.98, ' ' + lbl, fontsize=8, va='top')
    ax[0].set_ylim(0, 1); ax[0].set_ylabel('mIoU (média móvel)')
    ax[1].set_ylabel('nós do GWR'); ax[1].set_xlabel('imagem (ordem de captura)')
    ax[0].legend(fontsize=8, ncol=4); ax[1].legend(fontsize=8)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def run(a):
    os.makedirs(a.out, exist_ok=True)
    rs = np.random.RandomState(a.seed)
    gwrSeg.rand.seed(a.seed)
    ds = R.DATASETS[a.dataset]
    lm, ignore = np.array(ds['landing']), ds.get('ignore', [])
    K, class_names, colors = 3, R.LANDING_CLASSES[:3], R.LANDING_COLORS[:3]

    items = []
    for v, lst in R.find_pairs(a.root):
        for ip, lp, stem in lst:
            items.append((v, stem, ip, lp))
    tr_idx = [i for i, it in enumerate(items) if a.train_image in it[1]]
    if len(tr_idx) != 1:
        sys.exit('imagem de treino "%s" não encontrada (ou ambígua)' % a.train_image)
    tr_idx = tr_idx[0]

    def load(i):
        v, stem, ip, lp = items[i]
        img, lab = R.load_frame(ip, lp, a.width, ds['colors'])
        lab = np.where(np.isin(lab, ignore), -1, lab)
        lab = np.where(lab >= 0, lm[np.clip(lab, 0, None)], -1)
        X, shape, (ys, xs) = gwrSeg.extract_features(img, a.patch, a.stride, a.features,
                                                     levels=a.levels)
        return img, lab, X, shape, ys, xs

    # ---------------- Treino: uma imagem, 1% dos pontos rotulados ----------
    img, lab, X, shape, ys, xs = load(tr_idx)
    norm = gwrSeg.FeatureNormalizer(a.pca).fit(X)  # sem rótulos
    X = norm.transform(X)
    y_grid = lab[ys][:, xs].reshape(-1)
    valid = np.flatnonzero(y_grid >= 0)
    n_lab = int(round(a.label_frac * len(y_grid)))
    pick = rs.choice(valid, n_lab, replace=False)
    y_tr = -np.ones(len(y_grid), int)
    y_tr[pick] = y_grid[pick]
    cnt = np.bincount(y_tr[pick], minlength=K)
    print('imagem de treino: %s %s | %d pontos da grade, %d rotulados %s' %
          (items[tr_idx][0], items[tr_idx][1], len(y_grid), n_lab,
           dict(zip(class_names, cnt.tolist()))))
    pr, pc = np.unravel_index(pick, shape)
    save_train_figure(img[ys][:, xs], lab[ys][:, xs], (pr, pc, y_tr[pick]), colors,
                      class_names, os.path.join(a.out, 'treino.png'),
                      'treino: %s %s' % items[tr_idx][:2])

    # Cada modelo é treinado uma vez; a versão congelada é uma cópia exata
    # da adaptativa, de modo que as duas partem do mesmo estado
    import copy
    models = []
    for cls, base in ((GWRModel, 'GWR'), (SOMModel, 'SOM'), (KMeansModel, 'k-means'),
                      (KNNModel, 'k-NN')):
        if a.methods and not any(n.startswith(base) for n in a.methods):
            continue
        t0 = time.time()
        m = cls(base + ('' if base == 'k-NN' else '-adapt'), K, a, True)
        m.fit(X, y_tr)
        print('  %-14s treinado em %.1f s (tamanho %d)' % (base, time.time() - t0, m.size()))
        models.append(m)
        if base != 'k-NN':
            c = copy.deepcopy(m)
            c.name, c.adapt_on = base + '-cong', False
            models.append(c)
        if base == 'GWR':
            for src in (m, c):
                d = copy.deepcopy(src)
                d.name, d.nov_rule = src.name + '+nov', True
                models.append(d)
    if a.methods:
        models = [m for m in models if m.name in a.methods]

    # ---------------- Avaliação: demais imagens, sem rótulos ---------------
    conf = {m.name: np.zeros((K, K), np.int64) for m in models}
    conf_v = {m.name: {} for m in models}
    rows, boundaries, grid = [], [], []
    prev = None
    order = [i for i in range(len(items)) if i != tr_idx]
    vids = [items[i][0] for i in order]
    grid_steps = set()
    for v in dict.fromkeys(vids):
        ii = [k for k, x in enumerate(vids) if x == v]
        grid_steps.add(ii[len(ii) // 2])
    for step, i in enumerate(order):
        video, stem = items[i][:2]
        if video != prev:
            boundaries.append((step, video))
            prev = video
        img, lab, X, shape, ys, xs = load(i)
        X = norm.transform(X)
        y = lab[ys][:, xs].reshape(-1)
        sub = rs.choice(len(X), min(a.adapt_samples, len(X)), replace=False)
        preds, mious, nov_map = {}, {}, None
        for m in models:
            t0 = time.time()
            p, nov = m.predict(X)
            t_inf = (time.time() - t0) * 1000
            C = R.confusion(y, p, K)
            conf[m.name] += C
            cv = conf_v[m.name].setdefault(video, np.zeros((K, K), np.int64))
            cv += C
            miou = float(np.nanmean(R.iou_from_conf(C))) if (y >= 0).any() else np.nan
            t0 = time.time()
            m.adapt(X[sub])  # só a imagem, sem rótulos
            rows.append(dict(step=step, video=video, frame=stem, method=m.name, miou=miou,
                             size=m.size(), infer_ms=t_inf, adapt_s=time.time() - t0,
                             novelty=float(nov.mean()) if nov is not None else None))
            preds[m.name], mious[m.name] = p.reshape(shape), miou
            if m.name == 'GWR-cong':
                nov_map = nov.reshape(shape)
        print('[%3d/%d] %s %s  %s' % (step + 1, len(order), video, stem,
                                      ' '.join('%s=%.2f' % (k, v) for k, v in mious.items())),
              flush=True)
        if step in grid_steps and nov_map is not None:
            sel = [n for n in ('GWR-cong', 'GWR-cong+nov', 'SOM-cong', 'k-means-cong', 'k-NN')
                   if n in preds]
            grid.append(dict(img=img[ys][:, xs], gt=lab[ys][:, xs],
                             preds={n: preds[n] for n in sel},
                             miou={n: mious[n] for n in sel}, novelty=nov_map,
                             title='%s\n%s (#%d)' % (video, stem, step)))

    # ---------------- Resultados -------------------------------------------
    summary = {'args': vars(a), 'train_image': items[tr_idx][:2],
               'labeled_points': dict(zip(class_names, cnt.tolist())), 'methods': {}}
    print('\nResultado (todas as %d imagens não vistas, sem rótulos):' % len(order))
    print('%-15s %6s %6s %6s %6s %8s %8s %5s %7s' % ('modelo', 'mIoU', 'pous', 'nao', 'agua',
                                                     'falsoPous', 'precPous', 'tam.', 'inf(ms)'))
    for m in models:
        iou = R.iou_from_conf(conf[m.name])
        r = [x for x in rows if x['method'] == m.name]
        summary['methods'][m.name] = dict(
            mIoU=float(np.nanmean(iou)),
            IoU_per_class=dict(zip(class_names, map(float, iou))),
            mIoU_per_video={v: float(np.nanmean(R.iou_from_conf(C))) for v, C in conf_v[m.name].items()},
            conf_per_video={v: C.tolist() for v, C in conf_v[m.name].items()},
            final_size=r[-1]['size'], mean_infer_ms=float(np.mean([x['infer_ms'] for x in r])),
            mean_adapt_s=float(np.mean([x['adapt_s'] for x in r])),
            **safety(conf[m.name]))
        if isinstance(m, GWRModel):
            summary['methods'][m.name]['growth'] = {k: int(v) for k, v in m.net['stats'].items()}
        s = summary['methods'][m.name]
        print('%-15s %6.3f %6.3f %6.3f %6.3f %8.3f %8.3f %5d %7.1f' % (
            m.name, s['mIoU'], *iou, s['false_landable'], s['landable_precision'],
            s['final_size'], s['mean_infer_ms']))
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(a.out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_curves(rows, [m.name for m in models], boundaries, os.path.join(a.out, 'curvas.png'),
                'Uma imagem de treino (%s), sem rótulos depois' % items[tr_idx][1])
    if grid:
        R.save_qualitative_grid(grid, colors, class_names, os.path.join(a.out, 'qual_grade.png'))
    print('Resultados em', os.path.abspath(a.out))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    p.add_argument('--dataset', default='sdd', choices=sorted(R.DATASETS))
    p.add_argument('--train-image', required=True, help='nome (ou parte) da imagem de treino')
    p.add_argument('--label-frac', type=float, default=0.01,
                   help='fração dos pontos da grade da imagem de treino que recebe rótulo')
    p.add_argument('--train-epochs', type=int, default=5)
    p.add_argument('--adapt-samples', type=int, default=2000,
                   help='pontos (sem rótulo) de cada imagem usados na adaptação')
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--features', default='filters', choices=['stats', 'filters', 'pyramid'])
    p.add_argument('--levels', type=int, default=4)
    p.add_argument('--pca', type=int, default=12)
    p.add_argument('--patch', type=int, default=2)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--protos', type=int, default=30,
                   help='calibração do GWR; 30 dá ~100 nós após o treino, a mesma '
                        'capacidade do SOM 10x10 e do k-means com 100 grupos')
    p.add_argument('--max-nodes', type=int, default=600)
    p.add_argument('--som-shape', type=int, nargs=2, default=[10, 10])
    p.add_argument('--knn-k', type=int, default=5)
    p.add_argument('--methods', nargs='*')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='resultados_uma_imagem')
    return p.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
