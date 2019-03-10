#!/usr/bin/env python3

import argparse
import json
import spacy
import os
import pysrt
from collections import namedtuple
from termcolor import colored
from tqdm import tqdm

from nlp.translate import GoogleTranslator, DictionaryTranslator
from nlp.word_vectors import get_word_vectors


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_sub_path',
                        help='Path to source language SRT file.')
    parser.add_argument('dst_sub_path',
                        help='Path to destination language SRT file.')
    parser.add_argument('-s', dest='src_lang', required=True)
    parser.add_argument('-d', dest='dst_lang', default='en')
    parser.add_argument('-o', dest='out_file', required=True,
                        help='Processed captions file')
    parser.add_argument('-t', '--threshold', type=int, default=500,
                        help='Alignment tolerance in milliseconds')
    return parser.parse_args()


Caption = namedtuple('Caption', ['start', 'end', 'data'])
AlignedToken = namedtuple('AlignedToken', ['token', 'aligned_tokens'])


def load_srt(fname):
    lines = []
    for s in pysrt.open(fname):
        lines.append(Caption(
            start=s.start.ordinal, end=s.end.ordinal, data=s.text))
    return lines


def process_with_spacy(captions, lang):
    nlp = spacy.load(lang, disable=['ner'])
    tagged = []
    for c in tqdm(captions, desc='Running "{}" tagger'.format(lang)):
        line = c.data.strip()
        if line != '':
            tokens = list(nlp(line))
        else:
            tokens = []
        tagged.append(c._replace(data=tokens))
    return tagged


def get_vocabulary(captions):
    s = set()
    for c in captions:
        s.update([t.text.lower() for t in c.data
                  if t.pos_ != 'PUNCT' and t.pos_ != 'SYM'])
    return s


def overlap(a, b):
    return min(a.end, b.end) - max(a.start, b.start)


DIST_THRESHOLD = 1


def align_captions(src_lines, dst_lines, src_wv, dst_wv, src_lang, dst_lang,
                   threshold, gtrans_src_dst, gtrans_dst_src):
    translator = DictionaryTranslator(src_lang, dst_lang)
    aligned_lines = [
        c._replace(data=[AlignedToken(t, []) for t in c.data])
        for c in src_lines
    ]
    for d in tqdm(dst_lines, desc='Aligning "{}" -> "{}"'.format(
                  src_lang, dst_lang)):
        cand_lines = filter(
            lambda al: overlap(al, d) >= -threshold,
            aligned_lines)
        cand_tokens = [t for c in cand_lines for t in c.data]
        for dst_token in d.data:
            if dst_token.pos_ == 'PUNCT':
                continue

            dst_token_text = dst_token.text.lower()
            dst_token_vec = dst_wv.get(dst_token_text)

            max_cand = None
            max_sim = -1
            for cand_token in cand_tokens:
                if cand_token.token.pos_ == 'PUNCT':
                    continue

                if any(al_token.text.lower() == dst_token_text
                       for al_token in cand_token.aligned_tokens):
                    continue

                cand_token_text = cand_token.token.text.lower()
                cand_token_trans = translator.translate(cand_token_text)
                if dst_token_text in cand_token_trans:
                    sim = 1
                elif dst_token_text == gtrans_src_dst.get(cand_token_text):
                    sim = 1
                elif gtrans_dst_src.get(dst_token_text) == cand_token_text:
                    sim = 1
                else:
                    cand_token_vec = src_wv.get(cand_token_text)
                    if dst_token_vec is None or cand_token_vec is None:
                        sim = 0
                    else:
                        sim = dst_token_vec.dot(cand_token_vec)
                if sim > max_sim:
                    max_cand = cand_token
                    max_sim = sim
            if max_cand is not None and max_sim >= 0.3:
                max_cand.aligned_tokens.append(dst_token)
    return aligned_lines


def format_ms(t):
    ms = t % 1000
    s = int(t / 1000) % 60
    m = int(t / 60000) % 60
    h = int(t / 3600000)
    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(h, m, s, ms)


DEFAULT_COLORS = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']


def print_aligned_captions(lines, dst_lines):
    color_idx = 0
    for l in lines:
        print(format_ms(l.start), '-->', format_ms(l.end))
        tokens = []
        for t in l.data:
            if len(t.aligned_tokens) > 0:
                color = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
                tokens.append(colored(t.token.text, color))
                tokens.append(colored('[', color))
                for at in t.aligned_tokens:
                    tokens.append(colored(at.text, color))
                tokens.append(colored(']', color))
                color_idx += 1
            else:
                tokens.append(t.token.text)
        print(' '.join(tokens))
        for d in dst_lines:
            if overlap(l, d) > 0:
                print(' '.join([t.text for t in d.data]))
        print()


def get_aligned_captions_as_json(lines):
    return [
        {
            'start': l.start,
            'end': l.end,
            'tokens': [{
                'src': t.token.text,
                'dst': [x.text for x in t.aligned_tokens]
            } for t in l.data]
        } for l in lines
    ]


def get_captions_as_json(lines):
    return [{
        'start': l.start,
        'end': l.end,
        'tokens': [t.text for t in l.data]
    } for l in lines]


def set_api_key(key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path


def google_translate_vocab(vocab, src_lang, dst_lang):
    translator = GoogleTranslator(src_lang, dst_lang)
    result = {}
    for v in tqdm(vocab, desc='GTrans "{}" -> "{}"'.format(
                  src_lang, dst_lang)):
        if v.strip() == '':
            continue
        try:
            t = translator.translate(v)
            result[v] = t
        except:
            print('Cannot translate:', v)
    return result


def main(src_sub_path, dst_sub_path, src_lang, dst_lang, out_file,
         threshold, api_key_path='key.json'):
    if dst_lang != 'en':
        raise NotImplementedError()

    src_lines = process_with_spacy(load_srt(src_sub_path), src_lang)
    src_vocab = get_vocabulary(src_lines)
    src_wv = get_word_vectors(src_vocab, src_lang)

    dst_lines = process_with_spacy(load_srt(dst_sub_path), dst_lang)
    dst_vocab = get_vocabulary(dst_lines)
    dst_wv = get_word_vectors(dst_vocab, dst_lang)

    if api_key_path is not None:
        set_api_key(api_key_path)

    gtrans_cache_path = os.path.join(out_file + '.google-cache')
    if os.path.exists(gtrans_cache_path):
        with open(gtrans_cache_path) as f:
            gtrans_src_dst, gtrans_dst_src = json.load(f)
        print('Loaded cached Google Translations')
    else:
        gtrans_src_dst = google_translate_vocab(src_vocab, src_lang, dst_lang)
        gtrans_dst_src = google_translate_vocab(dst_vocab, dst_lang, src_lang)
        with open(gtrans_cache_path, 'w') as f:
            json.dump([gtrans_src_dst, gtrans_dst_src], f)

    aligned_lines = align_captions(src_lines, dst_lines, src_wv, dst_wv,
                                   src_lang, dst_lang, threshold,
                                   gtrans_src_dst, gtrans_dst_src)

    print_aligned_captions(aligned_lines, dst_lines)

    with open(out_file, 'w') as f:
        json.dump({
            'src': get_captions_as_json(src_lines),
            'dst': get_captions_as_json(dst_lines),
            'align': get_aligned_captions_as_json(aligned_lines)
        }, f)


if __name__ == '__main__':
    main(**vars(get_args()))
