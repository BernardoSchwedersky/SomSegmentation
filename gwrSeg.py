# -*- coding: utf-8 -*-
"""
Grow When Required (GWR) para segmentação de imagens, com extensão
semi-supervisionada.

Base: Marsland, Shapiro & Nehmzow (2002), "A self-organising network that
grows when required", Neural Networks 15(8-9), 1041-1058.

Extensões semi-supervisionadas (proposta do artigo):
  1. Cada nó guarda uma distribuição de rótulos P_j e um contador de evidência
     n_j, atualizados pelas (poucas) amostras rotuladas.
  2. Propagação de rótulos pelo grafo do próprio GWR (label propagation
     harmônica ancorada pela evidência de cada nó).
  3. Crescimento guiado por conflito: um nó confiante que erra repetidamente
     para amostras rotuladas ganha um novo nó vizinho com o rótulo correto,
     refinando a fronteira entre classes.
  4. Duas incertezas separadas na inferência: novidade (distância ao nó
     vencedor) e entropia do rótulo do nó.

Segue o estilo de somSeg.py: a rede é um dicionário e as funções operam
sobre ele.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from progress.bar import Bar

# Fixa a seed aleatória
rand = np.random.RandomState(0)

# Valor dos pesos de posições livres/removidas, para nunca vencerem a busca
INATIVO = 1e6


# ---------------------------------------------------------------------------
# Criação da rede
# ---------------------------------------------------------------------------

def create_gwr(nInputs, n_classes=0, max_nodes=1000, init_samples=None,
               act_thr=0.35, hab_thr=0.1, eps_b=0.1, eps_n=0.01,
               tau_b=0.3, tau_n=0.1, kappa=1.05, max_age=50,
               dist_scale=1.0, label_rate=0.1, conflict_conf=0.8,
               conflict_min_hits=5, conflict_count=3, recycle=True):
    """Cria um novo GWR com dois nós.

    nInputs     : dimensão das entradas
    n_classes   : número de classes (0 = puramente não supervisionado)
    max_nodes   : capacidade máxima de nós
    init_samples: se fornecido, os dois nós iniciais são amostras dele
    act_thr     : limiar de atividade a_T (insere nó se a < a_T)
    hab_thr     : limiar de habituação h_T (só insere se o vencedor já
                  foi bem treinado, h_b < h_T)
    eps_b/eps_n : taxas de aprendizado do vencedor / vizinhos
    tau_b/tau_n : constantes de habituação do vencedor / vizinhos
    kappa       : constante da curva de habituação
    max_age     : idade máxima das arestas
    dist_scale  : escala da distância na atividade a = exp(-d / dist_scale)
                  (use calibrate_dist_scale para estimar)
    label_rate  : taxa de atualização da distribuição de rótulos dos nós
    conflict_*  : parâmetros do crescimento guiado por conflito
    recycle     : com a rede cheia, recicla o nó usado há mais tempo (de
                  preferência sem evidência rotulada) em vez de parar de crescer
    """
    K = max(int(n_classes), 0)
    gwr = {
        'W': np.full((max_nodes, nInputs), INATIVO, dtype=float),
        'h': np.ones(max_nodes),
        'active': np.zeros(max_nodes, dtype=bool),
        'A': np.full((max_nodes, max_nodes), -1, dtype=np.int32),  # idade das arestas (-1 = sem aresta)
        'P': np.full((max_nodes, max(K, 1)), 1.0 / max(K, 1)),     # distribuição de rótulos
        'Pp': None,                                                # rótulos após propagação
        'n': np.zeros(max_nodes),                                  # evidência rotulada
        'conflict': np.zeros(max_nodes, dtype=int),
        'wins': np.zeros(max_nodes),
        'last_win': np.zeros(max_nodes),                           # passo da última vitória (LRU)
        't': 0,             # contador de passos de treino
        'top': 0,           # maior posição usada + 1
        'n_classes': K,
        'stats': {'inserted_novelty': 0, 'inserted_conflict': 0, 'removed': 0},
        'cfg': dict(act_thr=act_thr, hab_thr=hab_thr, eps_b=eps_b, eps_n=eps_n,
                    tau_b=tau_b, tau_n=tau_n, kappa=kappa, max_age=max_age,
                    dist_scale=dist_scale, label_rate=label_rate,
                    conflict_conf=conflict_conf,
                    conflict_min_hits=conflict_min_hits,
                    conflict_count=conflict_count, recycle=recycle),
    }
    if init_samples is not None and len(init_samples) >= 2:
        idx = rand.choice(len(init_samples), 2, replace=False)
        w0 = np.asarray(init_samples)[idx].astype(float)
    else:
        w0 = rand.rand(2, nInputs)
    for w in w0:
        _new_node(gwr, w)
    _connect(gwr, 0, 1)
    return gwr


# ---------------------------------------------------------------------------
# Funções auxiliares do grafo
# ---------------------------------------------------------------------------

def _new_node(gwr, w, P=None, n=0.0):
    """Adiciona um nó na primeira posição livre. Retorna o índice ou -1."""
    max_nodes = gwr['W'].shape[0]
    if gwr['top'] < max_nodes:
        r = gwr['top']
        gwr['top'] += 1
    else:
        livres = np.flatnonzero(~gwr['active'])
        if livres.size == 0:
            if not gwr['cfg'].get('recycle', False):
                return -1
            # Recicla o nó menos usado recentemente, preferindo nós sem rótulo
            cand = np.flatnonzero(gwr['active'] & (gwr['n'] == 0))
            if cand.size == 0:
                cand = np.flatnonzero(gwr['active'])
            r = cand[np.argmin(gwr['last_win'][cand])]
            _remove_node(gwr, r)
            gwr['stats']['recycled'] = gwr['stats'].get('recycled', 0) + 1
        else:
            r = livres[0]
    gwr['W'][r] = w
    gwr['h'][r] = 1.0
    gwr['active'][r] = True
    gwr['A'][r, :] = -1
    gwr['A'][:, r] = -1
    K = gwr['P'].shape[1]
    gwr['P'][r] = P if P is not None else 1.0 / K
    gwr['n'][r] = n
    gwr['conflict'][r] = 0
    gwr['wins'][r] = 0
    gwr['last_win'][r] = gwr['t']
    return r


def _remove_node(gwr, i):
    gwr['active'][i] = False
    gwr['W'][i] = INATIVO
    gwr['A'][i, :] = -1
    gwr['A'][:, i] = -1
    gwr['stats']['removed'] += 1


def _connect(gwr, i, j):
    gwr['A'][i, j] = 0
    gwr['A'][j, i] = 0


def _disconnect(gwr, i, j):
    gwr['A'][i, j] = -1
    gwr['A'][j, i] = -1


def _neighbors(gwr, i):
    return np.flatnonzero(gwr['A'][i, :gwr['top']] >= 0)


def n_nodes(gwr):
    """Número de nós ativos."""
    return int(gwr['active'].sum())


def edges(gwr):
    """Lista de arestas (i, j) com i < j entre nós ativos."""
    top = gwr['top']
    ii, jj = np.nonzero(np.triu(gwr['A'][:top, :top] >= 0, k=1))
    return np.stack([ii, jj], axis=1)


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def _two_best(d2):
    if d2.shape[0] == 2:
        return (0, 1) if d2[0] <= d2[1] else (1, 0)
    i2 = np.argpartition(d2, 1)[:2]
    if d2[i2[0]] <= d2[i2[1]]:
        return int(i2[0]), int(i2[1])
    return int(i2[1]), int(i2[0])


def gwr_step(gwr, x, y=-1, semi=True, conflict=True):
    """Atualiza o GWR com um exemplo x (e rótulo y, -1 = não rotulado).

    Retorna a atividade do vencedor.
    """
    cfg = gwr['cfg']
    W, h, A, P = gwr['W'], gwr['h'], gwr['A'], gwr['P']
    top = gwr['top']

    diff = W[:top] - x
    d2 = np.einsum('ij,ij->i', diff, diff)
    b, s = _two_best(d2)
    db = np.sqrt(d2[b])
    # Marca b e s como recentes antes de qualquer inserção, para que a
    # reciclagem (rede cheia) nunca remova os nós envolvidos neste passo
    gwr['t'] += 1
    gwr['last_win'][b] = gwr['t']
    gwr['last_win'][s] = gwr['t']

    # Aresta entre vencedor e segundo colocado (cria ou zera a idade)
    _connect(gwr, b, s)

    a = np.exp(-db / cfg['dist_scale'])
    K = gwr['n_classes']
    labeled = semi and K > 0 and y is not None and y >= 0
    if labeled:
        onehot = np.zeros(K)
        onehot[y] = 1.0

    inserted = False
    # Crescimento por novidade: atividade baixa e vencedor já habituado
    if a < cfg['act_thr'] and h[b] < cfg['hab_thr']:
        r = _new_node(gwr, (W[b] + x) / 2.0, P=P[b].copy(), n=0.0)
        if r >= 0:
            _connect(gwr, r, b)
            _connect(gwr, r, s)
            _disconnect(gwr, b, s)
            gwr['stats']['inserted_novelty'] += 1
            inserted = True
            if labeled:
                P[r] += cfg['label_rate'] * (onehot - P[r])
                gwr['n'][r] += 1

    if not inserted:
        # Adaptação do vencedor e dos vizinhos, ponderada pela habituação
        W[b] += cfg['eps_b'] * h[b] * (x - W[b])
        nb = _neighbors(gwr, b)
        if nb.size:
            W[nb] += cfg['eps_n'] * h[nb, None] * (x - W[nb])

    # Habituação (forma discreta de Marsland et al.)
    h[b] += cfg['tau_b'] * (cfg['kappa'] * (1.0 - h[b]) - 1.0)
    nb = _neighbors(gwr, b)
    if nb.size:
        h[nb] += cfg['tau_n'] * (cfg['kappa'] * (1.0 - h[nb]) - 1.0)
    t = gwr['top']
    np.clip(h[:t], 0.0, 1.0, out=h[:t])

    # Envelhecimento e remoção de arestas antigas e nós isolados
    if nb.size:
        A[b, nb] += 1
        A[nb, b] += 1
        velhas = nb[A[b, nb] > cfg['max_age']]
        for j in velhas:
            _disconnect(gwr, b, j)
            if _neighbors(gwr, j).size == 0 and n_nodes(gwr) > 2:
                _remove_node(gwr, j)

    gwr['wins'][b] += 1

    # Parte semi-supervisionada
    if labeled and not inserted:
        pred = int(np.argmax(P[b]))
        confiante = (gwr['n'][b] >= cfg['conflict_min_hits']
                     and P[b, pred] >= cfg['conflict_conf'])
        if conflict and confiante and pred != y:
            gwr['conflict'][b] += 1
            if gwr['conflict'][b] >= cfg['conflict_count']:
                # Nó confiante errou repetidamente: cria um nó com o rótulo correto
                r = _new_node(gwr, x.copy(), P=onehot.copy(), n=1.0)
                if r >= 0:
                    _connect(gwr, r, b)
                    gwr['conflict'][b] = 0
                    gwr['stats']['inserted_conflict'] += 1
                    return a
        elif pred == y and gwr['conflict'][b] > 0:
            gwr['conflict'][b] -= 1
        P[b] += cfg['label_rate'] * (onehot - P[b])
        gwr['n'][b] += 1
    return a


def train_GWR(gwr, data, labels=None, epochs=1, semi=True, conflict=True,
              shuffle=True, show_progress=True):
    """Treina o GWR com um conjunto de exemplos (e rótulos opcionais, -1 = sem rótulo)."""
    data = np.asarray(data, dtype=float)
    if labels is None:
        labels = -np.ones(len(data), dtype=int)
    labels = np.asarray(labels, dtype=int)
    bar = Bar('Processing', max=epochs) if show_progress else None
    for epoch in range(epochs):
        order = rand.permutation(len(data)) if shuffle else np.arange(len(data))
        for i in order:
            gwr_step(gwr, data[i], labels[i], semi=semi, conflict=conflict)
        if bar:
            bar.next()
    if bar:
        bar.finish()
    gwr['Pp'] = None  # propagação precisa ser refeita
    return gwr


# ---------------------------------------------------------------------------
# Propagação de rótulos pelo grafo
# ---------------------------------------------------------------------------

def train_GNG(gwr, data, labels=None, epochs=1, lam=300, max_nodes=100,
              eps_b=0.05, eps_n=0.006, alpha=0.5, beta=0.0005, max_age=50,
              shuffle=True):
    """Growing Neural Gas (Fritzke, 1995) sobre a mesma estrutura do GWR.

    Diferente do GWR, o GNG cresce em intervalos fixos (a cada `lam` passos),
    inserindo um nó entre o nó de maior erro acumulado e seu vizinho de maior
    erro, até `max_nodes`. A rotulagem semi-supervisionada (P, n), a
    propagação de rótulos e a novidade são as mesmas do GWR, de modo que a
    comparação isola o mecanismo de crescimento.
    """
    data = np.asarray(data, dtype=float)
    if labels is None:
        labels = -np.ones(len(data), dtype=int)
    labels = np.asarray(labels, dtype=int)
    cfg = gwr['cfg']
    K = gwr['n_classes']
    err = gwr.setdefault('err', np.zeros(gwr['W'].shape[0]))
    W, A, P = gwr['W'], gwr['A'], gwr['P']
    step = gwr.setdefault('gng_step', 0)
    for _ in range(epochs):
        order = rand.permutation(len(data)) if shuffle else np.arange(len(data))
        for i in order:
            x, y = data[i], labels[i]
            top = gwr['top']
            diff = W[:top] - x
            d2 = np.einsum('ij,ij->i', diff, diff)
            b, s = _two_best(d2)
            err[b] += d2[b]
            W[b] += eps_b * (x - W[b])
            nb = _neighbors(gwr, b)
            if nb.size:
                W[nb] += eps_n * (x - W[nb])
                A[b, nb] += 1
                A[nb, b] += 1
            _connect(gwr, b, s)
            nb = _neighbors(gwr, b)
            for j in nb[A[b, nb] > max_age]:
                _disconnect(gwr, b, j)
                if _neighbors(gwr, j).size == 0 and n_nodes(gwr) > 2:
                    _remove_node(gwr, j)
            if K > 0 and y >= 0:
                onehot = np.zeros(K)
                onehot[y] = 1.0
                P[b] += cfg['label_rate'] * (onehot - P[b])
                gwr['n'][b] += 1
            step += 1
            if step % lam == 0 and n_nodes(gwr) < max_nodes:
                act = np.flatnonzero(gwr['active'][:gwr['top']])
                q = act[np.argmax(err[act])]
                nq = _neighbors(gwr, q)
                if nq.size:
                    f = nq[np.argmax(err[nq])]
                    r = _new_node(gwr, (W[q] + W[f]) / 2.0, P=(P[q] + P[f]) / 2.0)
                    if r >= 0:
                        _disconnect(gwr, q, f)
                        _connect(gwr, r, q)
                        _connect(gwr, r, f)
                        err[q] *= alpha
                        err[f] *= alpha
                        err[r] = err[q]
                        gwr['stats']['inserted_novelty'] += 1
            err[:gwr['top']] *= (1.0 - beta)
    gwr['gng_step'] = step
    gwr['Pp'] = None
    return gwr


def fill_uninformed(gwr, tol=1e-3):
    """Nós cuja distribuição propagada continua (quase) uniforme, isto é, sem
    nenhum caminho até nós rotulados, recebem a distribuição do nó informado
    mais próximo no espaço de pesos. Evita que o empate (argmax de uma
    distribuição uniforme) atribua sempre a classe 0."""
    K = gwr['n_classes']
    if K == 0 or gwr['Pp'] is None:
        return gwr
    idx = np.flatnonzero(gwr['active'][:gwr['top']])
    Pp = gwr['Pp'][idx]
    informed = (Pp.max(1) - Pp.min(1)) > tol
    if informed.all() or not informed.any():
        return gwr
    Wi, Wu = gwr['W'][idx[informed]], gwr['W'][idx[~informed]]
    d2 = (Wu ** 2).sum(1)[:, None] + (Wi ** 2).sum(1)[None] - 2 * Wu @ Wi.T
    gwr['Pp'][idx[~informed]] = Pp[informed][np.argmin(d2, 1)]
    return gwr


def label_propagation(gwr, n0=5.0, iters=30, sigma2=None):
    """Propaga os rótulos dos nós com evidência para os demais pelo grafo do GWR.

    Para cada nó j:  P_j <- lambda_j * P_j^0 + (1 - lambda_j) * sum_k w_jk P_k
    com lambda_j = n_j / (n_j + n0) (nós com muita evidência ficam ancorados)
    e w_jk = exp(-||w_j - w_k||^2 / sigma2) normalizado nas arestas.
    """
    top = gwr['top']
    K = gwr['P'].shape[1]
    Pp = gwr['P'].copy()
    idx = np.flatnonzero(gwr['active'][:top])
    if gwr['n_classes'] == 0 or idx.size == 0:
        gwr['Pp'] = Pp
        return gwr
    adj = gwr['A'][np.ix_(idx, idx)] >= 0
    Wi = gwr['W'][idx]
    sq = (Wi ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Wi @ Wi.T, 0.0)
    if sigma2 is None:
        sigma2 = np.median(D2[adj]) if adj.any() else 1.0
        sigma2 = max(sigma2, 1e-12)
    Wg = np.where(adj, np.exp(-D2 / sigma2), 0.0)
    deg = Wg.sum(1, keepdims=True)
    T = np.divide(Wg, deg, out=np.zeros_like(Wg), where=deg > 0)

    P0 = gwr['P'][idx]
    n = gwr['n'][idx]
    lam = (n / (n + n0))[:, None]
    has_nb = (deg[:, 0] > 0)[:, None]
    Q = P0.copy()
    for _ in range(iters):
        viz = T @ Q
        Q = np.where(has_nb, lam * P0 + (1 - lam) * viz, P0)
    # Nós sem evidência e sem vizinhos rotulados continuam uniformes
    Q /= np.maximum(Q.sum(1, keepdims=True), 1e-12)
    Pp[idx] = Q
    gwr['Pp'] = Pp
    return gwr


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

def find_BMUs(gwr, X, batch=8192):
    """Nó vencedor e distância para cada linha de X (vetorizado)."""
    X = np.asarray(X, dtype=float)
    idx = np.flatnonzero(gwr['active'][:gwr['top']])
    Wi = gwr['W'][idx]
    w2 = (Wi ** 2).sum(1)
    bmu = np.empty(len(X), dtype=int)
    dist = np.empty(len(X))
    for s in range(0, len(X), batch):
        xb = X[s:s + batch]
        d2 = (xb ** 2).sum(1)[:, None] + w2[None, :] - 2.0 * xb @ Wi.T
        j = np.argmin(d2, axis=1)
        bmu[s:s + batch] = idx[j]
        dist[s:s + batch] = np.sqrt(np.maximum(d2[np.arange(len(xb)), j], 0.0))
    return bmu, dist


def predict_GWR(gwr, X, use_propagation=True):
    """Classifica X. Retorna dicionário com bmu, rótulo, novidade e entropia."""
    bmu, dist = find_BMUs(gwr, X)
    out = {'bmu': bmu, 'dist': dist,
           'novelty': 1.0 - np.exp(-dist / gwr['cfg']['dist_scale'])}
    if gwr['n_classes'] > 0:
        P = gwr['Pp'] if (use_propagation and gwr['Pp'] is not None) else gwr['P']
        Pb = P[bmu]
        out['label'] = np.argmax(Pb, axis=1)
        out['prob'] = Pb
        out['entropy'] = -(Pb * np.log(np.clip(Pb, 1e-12, 1))).sum(1) / np.log(gwr['n_classes'])
    return out


def evaluate_GWR(gwr, im, patch_size=2, stride=1, mode='patch'):
    """Equivalente ao evaluate_SOM: devolve o mapa de nós vencedores da imagem."""
    X, shape, _ = extract_features(im, patch_size, stride, mode)
    bmu, _ = find_BMUs(gwr, X)
    return bmu.reshape(shape)


def calibrate_dist_scale(X, n_protos=100, n=5000, quantile=0.9):
    """Estima dist_scale a partir do erro de quantização de ~n_protos
    protótipos (k-means++): quantil da distância de cada amostra até o
    protótipo mais próximo.

    Com act_thr = exp(-1) (padrão ~0.35), o GWR insere nós quando a entrada
    está mais longe que essa escala. O quantil alto (0.9) evita que as caudas
    da distribuição façam a rede crescer indefinidamente: ela estabiliza num
    tamanho da ordem de n_protos para dados parecidos com X e volta a crescer
    quando aparecem dados realmente novos.
    """
    X = np.asarray(X, dtype=float)
    S = X[rand.choice(len(X), min(n, len(X)), replace=False)]
    # Inicialização k-means++ (rápida) para os protótipos
    protos = [S[rand.randint(len(S))]]
    d2 = ((S - protos[0]) ** 2).sum(1)
    for _ in range(1, min(n_protos, len(S))):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        c = S[rand.choice(len(S), p=p)]
        protos.append(c)
        d2 = np.minimum(d2, ((S - c) ** 2).sum(1))
    return float(np.sqrt(np.quantile(np.maximum(d2, 0.0), quantile)))


# ---------------------------------------------------------------------------
# Extração de características
# ---------------------------------------------------------------------------

def extract_features(img, patch_size=2, stride=1, mode='patch', levels=4):
    """Extrai uma amostra por posição de uma grade com passo `stride`.

    mode='patch': patch (2b+1)x(2b+1) de todos os canais, na mesma ordem de
                  prepare_dataset_multichanel (canal a canal), escalado para [0, 1]
                  e com bordas replicadas.
    mode='stats': média e desvio padrão locais em Lab mais gradiente médio de L
                  (7 dimensões), mais robusto a pequenas variações de textura.

    Retorna X (M, D), shape (Hs, Ws) e as coordenadas (ys, xs) da grade.
    """
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    b = int(patch_size)
    k = 2 * b + 1
    scale = 255.0 if img.max() > 1.5 else 1.0

    if mode == 'patch':
        f = img.astype(np.float32) / scale
        if b > 0:
            f = np.pad(f, ((b, b), (b, b), (0, 0)), mode='edge')
            win = sliding_window_view(f, (k, k), axis=(0, 1))  # (H, W, C, k, k)
            X = win[ys][:, xs].reshape(len(ys) * len(xs), C * k * k)
        else:
            X = f[ys][:, xs].reshape(-1, C)
    elif mode == 'stats':
        from scipy.ndimage import uniform_filter, sobel
        from skimage.color import rgb2lab
        rgb = img[:, :, :3].astype(np.float32) / scale
        lab = rgb2lab(rgb).astype(np.float32)
        lab[:, :, 0] /= 100.0
        lab[:, :, 1:] = lab[:, :, 1:] / 128.0
        size = max(k, 3)
        mean = np.stack([uniform_filter(lab[:, :, c], size) for c in range(3)], -1)
        sq = np.stack([uniform_filter(lab[:, :, c] ** 2, size) for c in range(3)], -1)
        std = np.sqrt(np.maximum(sq - mean ** 2, 0.0))
        L = lab[:, :, 0]
        grad = uniform_filter(np.hypot(sobel(L, 0), sobel(L, 1)), size)
        F = np.concatenate([mean, std * 2.0, grad[:, :, None]], axis=-1)
        X = F[ys][:, xs].reshape(len(ys) * len(xs), F.shape[-1])
    elif mode == 'filters':
        F, _ = filter_bank(img[:, :, :3].astype(np.float32) / scale)
        X = F[ys][:, xs].reshape(len(ys) * len(xs), F.shape[-1])
    elif mode == 'pyramid':
        F, _ = pyramid_features(img[:, :, :3].astype(np.float32) / scale, levels=levels)
        X = F[ys][:, xs].reshape(len(ys) * len(xs), F.shape[-1])
    elif mode in ('patch_pyr', 'filters_pyr'):
        rgb = img[:, :, :3].astype(np.float32) / scale
        ext = (lambda im: dense_patches(im, b)) if mode == 'patch_pyr' else (lambda im: filter_bank(im)[0])
        F = gaussian_pyramid_stack(rgb, ext, levels)
        X = F[ys][:, xs].reshape(len(ys) * len(xs), F.shape[-1])
    else:
        raise ValueError("mode inválido: %s" % mode)
    return np.ascontiguousarray(X, dtype=float), (len(ys), len(xs)), (ys, xs)


# ---------------------------------------------------------------------------
# Banco de filtros convolucionais
# ---------------------------------------------------------------------------

_GABOR_CACHE = {}


def _gabor_pair(lam, theta):
    """Par de Gabor em quadratura (par/ímpar), média zero e norma L1 unitária."""
    key = (lam, theta)
    if key not in _GABOR_CACHE:
        import cv2
        sigma = 0.56 * lam                 # ~1 oitava de largura de banda
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        ev = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 1.0, 0.0, cv2.CV_32F)
        od = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 1.0, np.pi / 2, cv2.CV_32F)
        ev -= ev.mean()
        od -= od.mean()
        _GABOR_CACHE[key] = (ev / np.abs(ev).sum(), od / np.abs(od).sum())
    return _GABOR_CACHE[key]


def filter_bank(rgb, color_sigmas=(1, 3, 8), gabor_lambdas=(4, 8, 16),
                n_orient=6, log_sigmas=(1, 2, 4), grad_sigmas=(1, 3),
                std_sigmas=(3, 8)):
    """Banco de filtros convolucionais aplicado à imagem (H, W, 3) em [0, 1].

    Grupos de resposta (todos convoluções seguidas de retificação e
    agregação local com gaussiana, no estilo dos textons):
      cor     : L, a, b suavizados por gaussianas em várias escalas (contexto)
      gabor   : energia de Gabor em quadratura (sqrt(par^2 + ímpar^2)) em
                `n_orient` orientações e 3 comprimentos de onda; para cada
                escala, guarda o máximo e a média sobre as orientações, o que
                torna a feature invariante à rotação (câmera nadir)
      log     : |Laplaciano da gaussiana| (manchas/blobs) em várias escalas
      grad    : magnitude do gradiente gaussiano
      desvio  : desvio padrão local de L (rugosidade) em duas escalas

    Retorna F (H, W, D) float32 e a lista de nomes das D features.
    """
    import cv2
    from skimage.color import rgb2lab
    lab = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
    lab[:, :, 0] /= 100.0
    lab[:, :, 1:] /= 128.0
    L = np.ascontiguousarray(lab[:, :, 0])

    def blur(x, s):
        return cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)

    feats, names = [], []
    for s in color_sigmas:
        for c, n in enumerate('Lab'):
            feats.append(blur(np.ascontiguousarray(lab[:, :, c]), s))
            names.append('cor_%s_s%g' % (n, s))
    for lam in gabor_lambdas:
        en = []
        for k in range(n_orient):
            ev, od = _gabor_pair(lam, np.pi * k / n_orient)
            re = cv2.filter2D(L, cv2.CV_32F, ev, borderType=cv2.BORDER_REFLECT)
            im = cv2.filter2D(L, cv2.CV_32F, od, borderType=cv2.BORDER_REFLECT)
            en.append(blur(np.sqrt(re * re + im * im), 0.5 * lam))
        en = np.stack(en, 0)
        feats += [en.max(0), en.mean(0)]
        names += ['gabor_max_l%g' % lam, 'gabor_media_l%g' % lam]
    for s in log_sigmas:
        lg = cv2.Laplacian(blur(L, s), cv2.CV_32F, ksize=3) * (s * s)
        feats.append(blur(np.abs(lg), 2 * s))
        names.append('log_s%g' % s)
    for s in grad_sigmas:
        g = blur(L, s)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        feats.append(blur(np.sqrt(gx * gx + gy * gy) * s, 2 * s))
        names.append('grad_s%g' % s)
    for s in std_sigmas:
        m = blur(L, s)
        feats.append(np.sqrt(np.maximum(blur(L * L, s) - m * m, 0.0)))
        names.append('desvio_s%g' % s)
    return np.stack(feats, -1).astype(np.float32), names


def dense_patches(img, b):
    """Patch (2b+1)x(2b+1) de todos os canais em cada pixel (bordas
    replicadas), na ordem canal a canal de prepare_dataset_multichanel.
    Retorna (H, W, C*(2b+1)^2) float32."""
    k = 2 * b + 1
    H, W, C = img.shape
    f = np.pad(img.astype(np.float32), ((b, b), (b, b), (0, 0)), mode='edge')
    win = sliding_window_view(f, (k, k), axis=(0, 1))  # (H, W, C, k, k)
    return np.ascontiguousarray(win.reshape(H, W, C * k * k))


def gaussian_pyramid_stack(rgb, extractor, levels=4):
    """Aplica o mesmo extrator em cada nível da pirâmide gaussiana (conv
    gaussiana + redução 2x2 entre níveis, como no encoder da U-Net), reamostra
    os mapas de volta à resolução original (bilinear) e concatena tudo
    (conexões de salto). O campo receptivo do extrator dobra a cada nível."""
    import cv2
    H, W = rgb.shape[:2]
    out, cur = [], rgb.astype(np.float32)
    for lv in range(levels):
        if min(cur.shape[:2]) < 8:
            break
        F = extractor(cur).astype(np.float32)
        if lv > 0:
            # cv2.resize aceita até 512 canais; redimensiona em blocos
            F = np.concatenate([cv2.resize(np.ascontiguousarray(F[:, :, i:i + 256]), (W, H),
                                           interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
                                for i in range(0, F.shape[2], 256)], -1)
        out.append(F)
        cur = cv2.pyrDown(cur)
    return np.concatenate(out, -1)


def pyramid_features(rgb, levels=4, n_orient=6):
    """Encoder em pirâmide, no estilo do encoder da U-Net.

    Em cada nível l um banco de filtros *pequeno e fixo* é aplicado à imagem
    do nível; entre níveis a imagem passa por convolução gaussiana e redução
    2x2 (cv2.pyrDown = conv + pooling). Como os núcleos não crescem, o campo
    receptivo efetivo dobra a cada nível (Gabor com lambda = 4, 8, 16, 32...),
    com custo que cai 4x por nível. Os mapas de cada nível são reamostrados
    (bilinear) de volta à resolução original e concatenados, como as conexões
    de salto da U-Net.

    Banco por nível (8 mapas): L, a, b suavizados (sigma 1); energia de Gabor
    lambda=4 (máximo e média sobre orientações, invariante à rotação);
    |LoG| sigma 1; gradiente sigma 1; desvio local de L sigma 2.

    Retorna F (H, W, 8 * levels) float32 e os nomes.
    """
    import cv2
    from skimage.color import rgb2lab
    H, W = rgb.shape[:2]
    lab = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
    lab[:, :, 0] /= 100.0
    lab[:, :, 1:] /= 128.0

    def blur(x, s):
        return cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)

    feats, names = [], []
    cur = lab
    for lv in range(levels):
        h, w = cur.shape[:2]
        if min(h, w) < 8:
            break
        L = np.ascontiguousarray(cur[:, :, 0])
        maps = [blur(np.ascontiguousarray(cur[:, :, c]), 1.0) for c in range(3)]
        en = []
        for k in range(n_orient):
            ev, od = _gabor_pair(4, np.pi * k / n_orient)
            re = cv2.filter2D(L, cv2.CV_32F, ev, borderType=cv2.BORDER_REFLECT)
            im = cv2.filter2D(L, cv2.CV_32F, od, borderType=cv2.BORDER_REFLECT)
            en.append(blur(np.sqrt(re * re + im * im), 2.0))
        en = np.stack(en, 0)
        maps += [en.max(0), en.mean(0)]
        g = blur(L, 1.0)
        maps.append(blur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), 2.0))
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        maps.append(blur(np.sqrt(gx * gx + gy * gy), 2.0))
        m = blur(L, 2.0)
        maps.append(np.sqrt(np.maximum(blur(L * L, 2.0) - m * m, 0.0)))
        M = np.stack(maps, -1)
        if lv > 0:  # "decoder": volta à resolução original (skip connection)
            M = cv2.resize(M, (W, H), interpolation=cv2.INTER_LINEAR)
        feats.append(M)
        names += ['n%d_%s' % (lv, n) for n in
                  ('L', 'a', 'b', 'gabor_max', 'gabor_media', 'log', 'grad', 'desvio')]
        cur = cv2.pyrDown(cur)  # conv gaussiana + redução 2x2
    return np.concatenate(feats, -1).astype(np.float32), names


class FeatureNormalizer:
    """Padronização (z-score) e PCA opcional, ajustados uma única vez com o
    primeiro frame (sem rótulos) e mantidos fixos, como num sistema embarcado.
    """

    def __init__(self, n_pca=0):
        self.n_pca = int(n_pca)
        self.fitted = False

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        if self.n_pca > 0 and self.n_pca < X.shape[1]:
            _, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            self.P = Vt[:self.n_pca].T
            self.explained = float((S[:self.n_pca] ** 2).sum() / (S ** 2).sum())
        else:
            self.P = None
        self.fitted = True
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        return Z @ self.P if self.P is not None else Z
