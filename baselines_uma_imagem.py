# -*- coding: utf-8 -*-
"""
Baselines para o protocolo "uma imagem" no cenário binário (pousável x não
pousável; água = não pousável).

Mesmo protocolo de experimento_uma_imagem.py: uma imagem de treino com 1% dos
pontos da grade rotulados, nenhuma anotação depois, mesmas features (banco de
filtros + z-score + PCA ajustados na imagem de treino).

Grupos de métodos:
  triviais        : tudo-pousavel, tudo-nao-pousavel, cor-logistica (regressão
                    logística só com a cor Lab dos 96 pontos)
  supervisionados : k-NN, SVM-RBF, RandomForest, LogReg (só os 96 pontos)
  semi-superv.    : LabelSpreading (grafo k-NN sobre a imagem de treino),
                    SelfTraining-SVM (pseudo-rótulos na imagem de treino),
                    k-NN-pseudo (memória k-NN que incorpora pseudo-rótulos
                    confiantes de cada imagem de teste: adaptação com
                    pseudo-rótulos, sem anotação)
  protótipos      : GWR, GNG, SOM, k-means (congelados)
  com rejeição    : GWR+nov e GNG+nov (limiar natural 1 - a_T), e, no mesmo
                    ponto de operação, k-NN+rej (distância aos 96 pontos),
                    SOM+rej, k-means+rej (distância ao protótipo), GMM+rej
                    (verossimilhança), OCSVM+kNN, IForest+kNN, LOF+kNN
                    (detector de anomalia decide conhecido/desconhecido, k-NN
                    classifica o conhecido). Rejeitado = não pousável.

Ponto de operação comum: cada método com rejeição rejeita, na imagem de
treino, a mesma fração de pontos que o limiar natural do GWR rejeita.
Curvas de segurança: o limiar varre quantis do escore na imagem de treino.

Métrica de aplicação (ponto de pouso): em cada imagem, escolhe-se o ponto da
região prevista como pousável mais distante de qualquer ponto não pousável
(ou da borda). Se essa distância é menor que o raio de segurança, o método
se abstém. Senão, o pouso é seguro se >= 95% do disco de raio r em volta do
ponto é pousável no GT.

Uso:
  python baselines_uma_imagem.py --root sdd_prep --train-image t013_572
"""

import argparse
import copy
import warnings
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwrSeg  # noqa: E402
import somSeg  # noqa: E402
import ruralscapes_gwr as R  # noqa: E402
from experimento_uma_imagem import label_units, nearest, safety, save_train_figure  # noqa: E402

POUS, NAO = 0, 1
warnings.filterwarnings('ignore', category=FutureWarning)
QUANTIS = [None, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]


# ---------------------------------------------------------------------------
# Modelos: fit(X, y, ctx) e predict(X, ctx) -> (rótulo, escore de
# desconhecimento ou None). ctx traz features extras (cor) e estado.
# ---------------------------------------------------------------------------

class Const:
    def __init__(self, name, c):
        self.name, self.c = name, c

    def fit(self, X, y, ctx):
        pass

    def predict(self, X, ctx):
        return np.full(len(X), self.c), None


class SK:
    """Classificador scikit-learn treinado só nos pontos rotulados."""
    def __init__(self, name, make, use_color=False):
        self.name, self.make, self.use_color = name, make, use_color

    def _x(self, X, ctx):
        return ctx['color'] if self.use_color else X

    def fit(self, X, y, ctx):
        m = y >= 0
        self.clf = self.make().fit(self._x(X, ctx)[m], y[m])

    def predict(self, X, ctx):
        return self.clf.predict(self._x(X, ctx)), None


class KNNRej:
    """k-NN nos 96 pontos; escore = distância média aos k vizinhos."""
    def __init__(self, name, k=5):
        self.name, self.k = name, k

    def fit(self, X, y, ctx):
        from sklearn.neighbors import KNeighborsClassifier
        m = y >= 0
        self.clf = KNeighborsClassifier(n_neighbors=self.k).fit(X[m], y[m])

    def predict(self, X, ctx):
        d, _ = self.clf.kneighbors(X)
        return self.clf.predict(X), d.mean(1)


class Detector:
    """Detector de anomalia (ajustado em todos os pontos da imagem de treino,
    sem rótulos) + k-NN nos 96 pontos para classificar."""
    def __init__(self, name, kind):
        self.name, self.kind = name, kind

    def fit(self, X, y, ctx):
        from sklearn.neighbors import KNeighborsClassifier
        m = y >= 0
        self.knn = KNeighborsClassifier(n_neighbors=5).fit(X[m], y[m])
        rs = np.random.RandomState(ctx['seed'])
        if self.kind == 'ocsvm':
            from sklearn.svm import OneClassSVM
            sub = rs.choice(len(X), min(3000, len(X)), replace=False)
            self.det = OneClassSVM(nu=0.05, gamma='scale').fit(X[sub])
            self.score = lambda Z: -self.det.decision_function(Z)
        elif self.kind == 'iforest':
            from sklearn.ensemble import IsolationForest
            self.det = IsolationForest(n_estimators=200, random_state=ctx['seed']).fit(X)
            self.score = lambda Z: -self.det.score_samples(Z)
        elif self.kind == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            self.det = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X)
            self.score = lambda Z: -self.det.score_samples(Z)

    def predict(self, X, ctx):
        return self.knn.predict(X), self.score(X)


class GMMRej:
    """Mistura gaussiana na imagem de treino; cada componente recebe o rótulo
    da maioria dos pontos rotulados (ou do componente rotulado mais próximo);
    escore = -log-verossimilhança."""
    def __init__(self, name, n=20):
        self.name, self.n = name, n

    def fit(self, X, y, ctx):
        from sklearn.mixture import GaussianMixture
        self.g = GaussianMixture(self.n, covariance_type='full', reg_covar=1e-3,
                                 random_state=ctx['seed']).fit(X)
        m = y >= 0
        self.lab = label_units(self.g.means_, self.g.predict(X[m]), y[m], 2)

    def predict(self, X, ctx):
        return self.lab[self.g.predict(X)], -self.g.score_samples(X)


class Proto:
    """SOM ou k-means congelado; escore = distância ao protótipo vencedor."""
    def __init__(self, name, kind, shape=(10, 10)):
        self.name, self.kind, self.shape = name, kind, shape

    def fit(self, X, y, ctx):
        if self.kind == 'som':
            sx, sy = self.shape
            som = somSeg.create_map(sx, sy, X.shape[1])
            idx = gwrSeg.rand.choice(len(X), sx * sy, replace=False)
            som['weights'] = X[idx].reshape(sx, sy, -1).astype(float)
            som = somSeg.train_SOM(som, X.copy(), learn_rate=0.1, radius_sq=4,
                                   lr_decay=0.5, radius_decay=0.5, epochs=ctx['epochs'])
            self.W = som['weights'].reshape(-1, X.shape[1])
        else:
            from sklearn.cluster import MiniBatchKMeans
            self.W = MiniBatchKMeans(n_clusters=int(np.prod(self.shape)), random_state=ctx['seed'],
                                     n_init=3, batch_size=1024).fit(X).cluster_centers_
        m = y >= 0
        self.lab = label_units(self.W, nearest(self.W, X[m]), y[m], 2)

    def predict(self, X, ctx):
        u = nearest(self.W, X)
        d = np.sqrt(np.maximum(((X - self.W[u]) ** 2).sum(1), 0))
        return self.lab[u], d


class Graph:
    """GWR ou GNG (mesma rotulagem, propagação e novidade; muda o crescimento)."""
    def __init__(self, name, kind):
        self.name, self.kind = name, kind

    def fit(self, X, y, ctx):
        a = ctx['args']
        scale = gwrSeg.calibrate_dist_scale(X, n_protos=a.protos)
        self.net = gwrSeg.create_gwr(X.shape[1], n_classes=2, max_nodes=a.max_nodes,
                                     init_samples=X, dist_scale=scale)
        if self.kind == 'gwr':
            gwrSeg.train_GWR(self.net, X, y, epochs=ctx['epochs'], show_progress=False)
        else:
            steps = ctx['epochs'] * len(X)
            gwrSeg.train_GNG(self.net, X, y, epochs=ctx['epochs'], max_nodes=a.gng_nodes,
                             lam=max(1, steps // (2 * a.gng_nodes)))
        gwrSeg.label_propagation(self.net)
        gwrSeg.fill_uninformed(self.net)

    def predict(self, X, ctx):
        o = gwrSeg.predict_GWR(self.net, X, use_propagation=True)
        return o['label'], o['novelty']

    def natural_thr(self):
        return 1.0 - self.net['cfg']['act_thr']


class LabelSpread:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y, ctx):
        from sklearn.semi_supervised import LabelSpreading
        rs = np.random.RandomState(ctx['seed'])
        # usa todos os rotulados e uma amostra dos não rotulados da imagem de treino
        m = np.flatnonzero(y >= 0)
        u = rs.choice(np.flatnonzero(y < 0), min(4000, int((y < 0).sum())), replace=False)
        idx = np.concatenate([m, u])
        ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2, max_iter=100)
        ls.fit(X[idx], y[idx])
        # Extensão indutiva: os pontos da imagem de treino ficam com os rótulos
        # transduzidos e as imagens novas são classificadas por k-NN sobre eles.
        # (O predict do LabelSpreading devolve NaN quando nenhum vizinho tem
        # peso, e o argmax de NaN cai na classe 0 = pousável.)
        from sklearn.neighbors import KNeighborsClassifier
        self.clf = KNeighborsClassifier(n_neighbors=5).fit(X[idx], ls.transduction_)

    def predict(self, X, ctx):
        return self.clf.predict(X), None


class SelfTrain:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y, ctx):
        from sklearn.semi_supervised import SelfTrainingClassifier
        from sklearn.svm import SVC
        rs = np.random.RandomState(ctx['seed'])
        m = np.flatnonzero(y >= 0)
        u = rs.choice(np.flatnonzero(y < 0), min(4000, int((y < 0).sum())), replace=False)
        idx = np.concatenate([m, u])
        self.clf = SelfTrainingClassifier(SVC(probability=True, gamma='scale', random_state=ctx['seed']),
                                          threshold=0.9, max_iter=10)
        self.clf.fit(X[idx], y[idx])

    def predict(self, X, ctx):
        return self.clf.predict(X), None


class KNNPseudo:
    """Memória k-NN que começa com os 96 pontos rotulados e, a cada imagem de
    teste, incorpora até `per_img` pontos cuja previsão é unânime entre os k
    vizinhos (pseudo-rótulos). Os 96 pontos originais nunca saem; os
    pseudo-rótulos ficam numa fila de tamanho máximo `cap`."""
    def __init__(self, name, k=5, per_img=100, cap=4000):
        self.name, self.k, self.per_img, self.cap = name, k, per_img, cap

    def fit(self, X, y, ctx):
        m = y >= 0
        self.X0, self.y0 = X[m], y[m]
        self.Xp = np.zeros((0, X.shape[1]))
        self.yp = np.zeros(0, int)
        self.rs = np.random.RandomState(ctx['seed'])
        self._refit()

    def _refit(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.clf = KNeighborsClassifier(n_neighbors=self.k).fit(
            np.vstack([self.X0, self.Xp]), np.concatenate([self.y0, self.yp]))

    def predict(self, X, ctx):
        proba = self.clf.predict_proba(X)
        lab = self.clf.classes_[proba.argmax(1)]
        conf = np.flatnonzero(proba.max(1) >= 1.0)
        if conf.size:
            take = []
            for c in np.unique(lab[conf]):  # equilibra as classes
                cc = conf[lab[conf] == c]
                take.append(self.rs.choice(cc, min(self.per_img // 2, cc.size), replace=False))
            take = np.concatenate(take)
            self.Xp = np.vstack([self.Xp, X[take]])[-self.cap:]
            self.yp = np.concatenate([self.yp, lab[take]])[-self.cap:]
            self._refit()
        return lab, None


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= r * r


def landing_point(pred, gt, r):
    """Retorna 'seguro', 'inseguro' ou 'abstem' e a posição escolhida."""
    M = np.pad(pred == POUS, 1, constant_values=False)  # borda = desconhecido
    dt = distance_transform_edt(M)[1:-1, 1:-1]
    if dt.max() < r:
        return 'abstem', None
    py, px = np.unravel_index(np.argmax(dt), dt.shape)
    D = disk(r)
    y0, x0 = py - r, px - r
    g = gt[max(y0, 0):py + r + 1, max(x0, 0):px + r + 1]
    d = D[max(0, -y0):D.shape[0] - max(0, py + r + 1 - gt.shape[0]),
          max(0, -x0):D.shape[1] - max(0, px + r + 1 - gt.shape[1])]
    v = g[d]
    v = v[v >= 0]
    ok = v.size > 0 and (v == POUS).mean() >= 0.95
    return ('seguro' if ok else 'inseguro'), (py, px)


def gt_has_safe_area(gt, r):
    M = np.pad(gt == POUS, 1, constant_values=False)
    return distance_transform_edt(M).max() >= r


def iou2(C):
    return R.iou_from_conf(np.asarray(C))


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def build_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    return [
        ('trivial', Const('tudo-pousavel', POUS)),
        ('trivial', Const('tudo-nao-pousavel', NAO)),
        ('trivial', SK('cor-logistica', lambda: LogisticRegression(max_iter=1000, class_weight='balanced'),
                       use_color=True)),
        ('supervisionado', SK('k-NN', lambda: KNeighborsClassifier(5))),
        ('supervisionado', SK('SVM-RBF', lambda: SVC(gamma='scale', class_weight='balanced'))),
        ('supervisionado', SK('RandomForest', lambda: RandomForestClassifier(200, random_state=0,
                                                                             class_weight='balanced'))),
        ('supervisionado', SK('LogReg', lambda: LogisticRegression(max_iter=1000, class_weight='balanced'))),
        ('semi', LabelSpread('LabelSpreading')),
        ('semi', SelfTrain('SelfTraining-SVM')),
        ('semi', KNNPseudo('k-NN-pseudo')),
        ('protótipos', Graph('GWR', 'gwr')),
        ('protótipos', Graph('GNG', 'gng')),
        ('protótipos', Proto('SOM', 'som')),
        ('protótipos', Proto('k-means', 'kmeans')),
        ('rejeição', KNNRej('k-NN+rej')),
        ('rejeição', GMMRej('GMM')),
        ('rejeição', Detector('OCSVM+kNN', 'ocsvm')),
        ('rejeição', Detector('IForest+kNN', 'iforest')),
        ('rejeição', Detector('LOF+kNN', 'lof')),
    ]


def run(a):
    os.makedirs(a.out, exist_ok=True)
    rs = np.random.RandomState(a.seed)
    gwrSeg.rand.seed(a.seed)
    K, class_names, colors = 2, R.LANDING_CLASSES[:2], R.LANDING_COLORS[:2]
    from skimage.color import rgb2lab

    def items_of(root, videos=None):
        return [(v, stem, ip, lp) for v, lst in R.find_pairs(root, videos) for ip, lp, stem in lst]

    def loader(ds_name):
        ds = R.DATASETS[ds_name]
        lm, ignore = np.array(ds['landing']), ds.get('ignore', [])

        def load(item):
            v, stem, ip, lp = item
            img, lab = R.load_frame(ip, lp, a.width, ds['colors'])
            lab = np.where(np.isin(lab, ignore), -1, lab)
            lab = np.where(lab >= 0, lm[np.clip(lab, 0, None)], -1)
            lab = np.where(lab >= 1, NAO, lab)  # água e céu -> não pousável
            X, shape, (ys, xs) = gwrSeg.extract_features(img, a.patch, a.stride, a.features,
                                                         levels=a.levels)
            color = rgb2lab(img[ys][:, xs].astype(np.float32) / 255.0).reshape(-1, 3) / [100, 128, 128]
            return img, lab, X, color, shape, ys, xs
        return load

    items = items_of(a.root)
    tr = [i for i, it in enumerate(items) if a.train_image in it[1]]
    assert len(tr) == 1, 'imagem de treino não encontrada'
    tr = tr[0]
    load_train = loader(a.dataset)
    # Teste: as demais imagens do mesmo dataset, ou (extrapolação) as de
    # outro dataset passado em --test-root / --test-dataset
    if a.test_root:
        test_items = items_of(a.test_root, a.test_videos)
        load_test = loader(a.test_dataset)
        print('extrapolação: treino em %s (%s), teste em %s (%s, %d imagens)' %
              (a.root, a.dataset, a.test_root, a.test_dataset, len(test_items)))
    else:
        test_items = [it for i, it in enumerate(items) if i != tr]
        load_test = load_train

    # ---------------- treino ----------------
    img, lab, X, color, shape, ys, xs = load_train(items[tr])
    norm = gwrSeg.FeatureNormalizer(a.pca).fit(X)
    X = norm.transform(X)
    yg = lab[ys][:, xs].reshape(-1)
    valid = np.flatnonzero(yg >= 0)
    pick = rs.choice(valid, int(round(a.label_frac * len(yg))), replace=False)
    y_tr = -np.ones(len(yg), int)
    y_tr[pick] = yg[pick]
    cnt = np.bincount(y_tr[pick], minlength=K)
    print('treino %s %s: %d rotulados %s' % (items[tr][0], items[tr][1], len(pick),
                                            dict(zip(class_names, cnt.tolist()))))
    pr, pc = np.unravel_index(pick, shape)
    save_train_figure(img[ys][:, xs], lab[ys][:, xs], (pr, pc, y_tr[pick]), colors, class_names,
                      os.path.join(a.out, 'treino.png'), 'treino: %s %s' % items[tr][:2])

    ctx = dict(color=color, seed=a.seed, epochs=a.train_epochs, args=a)
    groups, models = {}, []
    for grp, m in build_models():
        if a.methods and m.name not in a.methods:
            continue
        t0 = time.time()
        m.fit(X, y_tr, ctx)
        groups[m.name] = grp
        models.append(m)
        print('  %-18s treinado em %5.1f s' % (m.name, time.time() - t0))

    # escores na imagem de treino -> limiares (curvas e ponto de operação)
    train_scores = {}
    for m in models:
        if isinstance(m, KNNPseudo):
            continue
        _, sc = m.predict(X, ctx)
        if sc is not None:
            train_scores[m.name] = sc
    # Ponto de operação comum: rejeita o que é mais estranho que a fração
    # --op-quantile dos pontos da imagem de treino (mesma regra para todos).
    # GWR e GNG aparecem também com o limiar natural (1 - a_T) como '+nov'.
    rej_frac_nat = {}
    op_thr, curve_thr = {}, {}
    for n, sc in train_scores.items():
        op_thr[n] = float(np.quantile(sc, a.op_quantile))
        curve_thr[n] = [np.inf if q is None else float(np.quantile(sc, q)) for q in QUANTIS]
    nat_thr = {m.name: m.natural_thr() for m in models if isinstance(m, Graph)}
    for n, t in nat_thr.items():
        rej_frac_nat[n] = float((train_scores[n] > t).mean())
        print('%s: limiar natural rejeita %.3f da imagem de treino' % (n, rej_frac_nat[n]))
        rej_names = {n: (n if n.endswith('+rej') or '+kNN' in n else n + '+rej') for n in train_scores}
    out_names = []
    for m in models:
        if m.name in train_scores and rej_names[m.name] == m.name:
            out_names.append(m.name)  # o nome já indica rejeição
        else:
            out_names.append(m.name)
            if m.name in train_scores:
                out_names.append(rej_names[m.name])
            if m.name in nat_thr:
                out_names.append(m.name + '+nov')

    conf = {n: np.zeros((K, K), np.int64) for n in out_names}
    conf_curve = {n: [np.zeros((K, K), np.int64) for _ in QUANTIS] for n in train_scores}
    landing = {n: {'seguro': 0, 'inseguro': 0, 'abstem': 0} for n in out_names}
    gt_safe = 0
    times = {n: [] for n in out_names}

    order = list(range(len(test_items)))
    for step, i in enumerate(order):
        img, lab, X, color, shape, ys, xs = load_test(test_items[i])
        X = norm.transform(X)
        ctx['color'] = color
        gt = lab[ys][:, xs]
        y = gt.reshape(-1)
        gt_safe += gt_has_safe_area(gt, a.safe_radius)
        for m in models:
            t0 = time.time()
            p, sc = m.predict(X, ctx)
            dt = (time.time() - t0) * 1000
            outs = []
            if m.name in train_scores:
                pr_rej = np.where(sc > op_thr[m.name], NAO, p)
                if rej_names[m.name] != m.name:
                    outs.append((m.name, p))
                outs.append((rej_names[m.name], pr_rej))
                if m.name in nat_thr:
                    outs.append((m.name + '+nov', np.where(sc > nat_thr[m.name], NAO, p)))
                for qi, thr in enumerate(curve_thr[m.name]):
                    conf_curve[m.name][qi] += R.confusion(y, np.where(sc > thr, NAO, p), K)
            else:
                outs.append((m.name, p))
            for n, pp in outs:
                conf[n] += R.confusion(y, pp, K)
                res, _ = landing_point(pp.reshape(shape), gt, a.safe_radius)
                landing[n][res] += 1
                times[n].append(dt)
        if step % 25 == 0 or step == len(order) - 1:
            print('[%3d/%d] %s %s' % (step + 1, len(order), *test_items[i][:2]), flush=True)

    # ---------------- resultados ----------------
    N = len(order)
    summary = dict(args=vars(a), train_image=items[tr][:2], labeled=dict(zip(class_names, cnt.tolist())),
                   op_quantile=a.op_quantile, natural_rej_frac_train=rej_frac_nat, n_images=N, gt_safe_area_frac=gt_safe / N,
                   quantis=[q for q in QUANTIS], methods={}, curves={})
    print('\n%-18s %-14s %6s %6s %6s %7s %7s %7s | %6s %6s %6s' % (
        'método', 'grupo', 'mIoU', 'IoUpou', 'IoUnao', 'cobPou', 'falsoP', 'precP', 'seguro', 'insegu', 'abstem'))
    for n in out_names:
        C = conf[n]
        iou = iou2(C)
        s = safety(C)
        cov = float(C[POUS, POUS] / max(C[POUS].sum(), 1))
        lp = {k: v / N for k, v in landing[n].items()}
        base = n.replace('+nov', '').replace('+rej', '')
        grp = groups.get(n, groups.get(base, 'rejeição'))
        if n.endswith(('+nov', '+rej')) or '+kNN' in n:
            grp = 'rejeição'
        summary['methods'][n] = dict(group=grp, mIoU=float(np.nanmean(iou)), IoU_pousavel=float(iou[0]),
                                     IoU_nao=float(iou[1]), landable_coverage=cov, **s,
                                     landing=lp, conf=C.tolist(),
                                     infer_ms=float(np.mean(times[n])) if times[n] else None)
        print('%-18s %-14s %6.3f %6.3f %6.3f %7.3f %7.3f %7.3f | %6.3f %6.3f %6.3f' % (
            n, grp, np.nanmean(iou), iou[0], iou[1], cov, s['false_landable'], s['landable_precision'],
            lp['seguro'], lp['inseguro'], lp['abstem']))
    print('imagens em que o GT tem área pousável com o raio de segurança: %.3f' % (gt_safe / N))
    for n in train_scores:
        pts = []
        for Cq in conf_curve[n]:
            Cq = np.asarray(Cq, float)
            pts.append(dict(coverage=float(Cq[0, 0] / max(Cq[0].sum(), 1)),
                            false_landable=float(Cq[1, 0] / max(Cq[1].sum(), 1))))
        summary['curves'][n] = pts
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    plot_curves(summary, os.path.join(a.out, 'curvas_seguranca.png'))
    print('Resultados em', os.path.abspath(a.out))


def plot_curves(summaries, path, title=None):
    """summaries: um summary ou lista (curvas médias entre imagens de treino)."""
    if isinstance(summaries, dict):
        summaries = [summaries]
    names = list(summaries[0]['curves'])
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for n in names:
        cov = np.mean([[p['coverage'] for p in s['curves'][n]] for s in summaries], 0)
        fl = np.mean([[p['false_landable'] for p in s['curves'][n]] for s in summaries], 0)
        lw = 2.4 if n in ('GWR', 'GNG') else 1.2
        ln, = ax.plot(fl, cov, '-o', ms=3, lw=lw, label=n)
        on = n if (n.endswith('+rej') or '+kNN' in n) else n + '+rej'
        if on in summaries[0]['methods']:
            ox = np.mean([s['methods'][on]['false_landable'] for s in summaries])
            oy = np.mean([s['methods'][on]['landable_coverage'] for s in summaries])
            ax.plot(ox, oy, '*', ms=13, color=ln.get_color(), markeredgecolor='k')
        if n + '+nov' in summaries[0]['methods']:
            ox = np.mean([s['methods'][n + '+nov']['false_landable'] for s in summaries])
            oy = np.mean([s['methods'][n + '+nov']['landable_coverage'] for s in summaries])
            ax.plot(ox, oy, 'D', ms=8, color=ln.get_color(), markeredgecolor='k')
    ax.set_xlabel('falso pousável (fração do terreno não pousável chamada de pousável) ↓')
    ax.set_ylabel('cobertura do pousável (fração do pousável recuperada) ↑')
    ax.set_title(title or 'Curvas de segurança (★ ponto de operação comum, ◆ limiar natural do GWR/GNG)',
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    p.add_argument('--dataset', default='sdd', choices=sorted(R.DATASETS))
    p.add_argument('--train-image', required=True)
    p.add_argument('--test-root', help='extrapolação: pasta de outro dataset para o teste')
    p.add_argument('--test-dataset', default='uavid', choices=sorted(R.DATASETS),
                   help='paleta/classes do dataset de teste')
    p.add_argument('--test-videos', nargs='*', help='sequências do dataset de teste')
    p.add_argument('--label-frac', type=float, default=0.01)
    p.add_argument('--train-epochs', type=int, default=5)
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--features', default='filters')
    p.add_argument('--levels', type=int, default=4)
    p.add_argument('--pca', type=int, default=12)
    p.add_argument('--patch', type=int, default=2)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--protos', type=int, default=30)
    p.add_argument('--max-nodes', type=int, default=600)
    p.add_argument('--gng-nodes', type=int, default=100, help='nós do GNG (capacidade igual à do GWR)')
    p.add_argument('--safe-radius', type=int, default=4,
                   help='raio de segurança do ponto de pouso, em pontos da grade (4 = 16 px a 480 px)')
    p.add_argument('--op-quantile', type=float, default=0.99,
                   help='ponto de operação comum: rejeita escores acima deste quantil da imagem de treino')
    p.add_argument('--methods', nargs='*')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='resultados_baselines')
    return p.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
