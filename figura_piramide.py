# -*- coding: utf-8 -*-
"""
Figura do encoder em pirâmide: para uma imagem, mostra em cada nível a
imagem reduzida e dois mapas de resposta (energia de Gabor máxima e
desvio local), já reamostrados para a resolução original, como entram no GWR.

Uso:
  python figura_piramide.py imagem.jpg --levels 6 --width 480 --out piramide.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwrSeg  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('imagem')
    p.add_argument('--levels', type=int, default=6)
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--out', default='piramide.png')
    a = p.parse_args(argv)

    import cv2
    im = Image.open(a.imagem).convert('RGB')
    h = int(round(im.height * a.width / im.width))
    rgb = np.asarray(im.resize((a.width, h), Image.BILINEAR)).astype(np.float32) / 255.0
    F, names = gwrSeg.pyramid_features(rgb, levels=a.levels)
    n_lv = F.shape[-1] // 8

    fig, ax = plt.subplots(3, n_lv, figsize=(2.6 * n_lv, 2.6 * 3 * h / a.width + 0.6), squeeze=False)
    cur = rgb
    for lv in range(n_lv):
        ax[0, lv].imshow(np.clip(cur, 0, 1), interpolation='nearest')
        ax[0, lv].set_title('nível %d  (%dx%d)\nGabor λ≈%d px' % (lv, cur.shape[1], cur.shape[0], 4 * 2 ** lv),
                            fontsize=8)
        for r, key in ((1, 'gabor_max'), (2, 'desvio')):
            m = F[:, :, names.index('n%d_%s' % (lv, key))]
            ax[r, lv].imshow(m, cmap='viridis', vmin=np.percentile(m, 1), vmax=np.percentile(m, 99))
        cur = cv2.pyrDown(cur)
    ax[0, 0].set_ylabel('imagem do nível', fontsize=9)
    ax[1, 0].set_ylabel('energia Gabor (máx.)', fontsize=9)
    ax[2, 0].set_ylabel('desvio local de L', fontsize=9)
    for x in ax.ravel():
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print('figura salva em', os.path.abspath(a.out))


if __name__ == '__main__':
    main()
