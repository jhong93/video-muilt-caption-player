import numpy as np
import os


FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_word_vectors(vocab, lang, normalize=True):
    print('Loading "{}" word vectors'.format(lang))
    wv_file = os.path.join(FILE_DIR, 'word_vectors',
                           'wiki.{}.align.vec'.format(lang))
    wv = {}
    with open(wv_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            token, data = line.split(' ', 1)
            if token in vocab:
                vec = np.array([float(x) for x in data.split(' ')])
                if normalize:
                    vec /= np.linalg.norm(vec)
                wv[token] = vec
    print('Loaded {} (of {} requested)'.format(len(wv), len(vocab)))
    return wv
