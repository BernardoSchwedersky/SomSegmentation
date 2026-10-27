# -*- coding: utf-8 -*-
"""
Prepara o Semantic Drone Dataset (TU Graz) para o experimento online.

O SDD não é um vídeo: são 400 fotos nadir (5 a 30 m) de 6000x4000 px. Mas o
EXIF guarda o horário de captura, então ordenar as fotos por ele reconstrói
a ordem real dos voos. Intervalos longos entre fotos separam voos distintos,
que o experimento trata como "sequências" (e as fronteiras como mudança de
domínio).

Entrada:  <src>/original_images/*.jpg e <src>/label_images_semantic/*.png
Saída:    <dst>/voo_XX/Images/tNNN_<id>.jpg e <dst>/voo_XX/Labels/tNNN_<id>.png
          (formato que o ruralscapes_gwr.py lê com --dataset sdd)

Uso:
  python prepare_sdd.py --src semantic_drone_dataset --dst sdd_prep --width 1500
"""

import argparse
import datetime as dt
import glob
import json
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def capture_time(path):
    """Horário de captura pelo EXIF (DateTimeOriginal ou DateTime)."""
    ex = Image.open(path).getexif()
    s = ex.get_ifd(0x8769).get(36867) or ex.get(306)  # DateTimeOriginal / DateTime
    return dt.datetime.strptime(s.strip(), '%Y:%m:%d %H:%M:%S') if s else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--width', type=int, default=1500, help='largura de saída das imagens')
    p.add_argument('--gap-min', type=float, default=30.0,
                   help='intervalo (min) entre fotos que separa dois voos')
    a = p.parse_args(argv)

    imgs = sorted(glob.glob(os.path.join(a.src, 'original_images', '*.jpg')))
    items = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(a.src, 'label_images_semantic', stem + '.png')
        if os.path.exists(lp):
            items.append((capture_time(ip), stem, ip, lp))
    sem_hora = [s for t, s, _, _ in items if t is None]
    if sem_hora:
        print('aviso: %d imagens sem horário no EXIF (vão para o fim)' % len(sem_hora))
    far = dt.datetime.max
    items.sort(key=lambda t: (t[0] or far, t[1]))

    # Separa voos por intervalos longos entre capturas
    voo, prev, info = 0, None, []
    for k, (t, stem, ip, lp) in enumerate(items):
        if prev is not None and t is not None and (t - prev).total_seconds() > a.gap_min * 60:
            voo += 1
        prev = t or prev
        d = os.path.join(a.dst, 'voo_%02d' % voo)
        os.makedirs(os.path.join(d, 'Images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'Labels'), exist_ok=True)
        name = 't%03d_%s' % (k, stem)
        im = Image.open(ip).convert('RGB')
        h = int(round(im.height * a.width / im.width))
        im.resize((a.width, h), Image.BILINEAR).save(os.path.join(d, 'Images', name + '.jpg'), quality=92)
        lab = Image.open(lp)
        lab.resize((a.width, h), Image.NEAREST).save(os.path.join(d, 'Labels', name + '.png'))
        info.append(dict(ordem=k, voo=voo, id=stem, hora=t.isoformat() if t else None))
        print('\r%d/%d' % (k + 1, len(items)), end='', flush=True)
    print()
    with open(os.path.join(a.dst, 'ordem_captura.json'), 'w') as f:
        json.dump(info, f, indent=1)
    for v in range(voo + 1):
        r = [x for x in info if x['voo'] == v]
        print('voo_%02d: %3d fotos, %s -> %s' % (v, len(r), r[0]['hora'], r[-1]['hora']))


if __name__ == '__main__':
    main()
