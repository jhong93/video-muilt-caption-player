#!/usr/bin/env python3

import argparse
import os
import pysrt

from nlp.translate import GoogleTranslator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_file')
    parser.add_argument('src_lang')
    parser.add_argument('dst_file')
    parser.add_argument('dst_lang')
    parser.add_argument('-k', '--api-key', dest='api_key_path')
    return parser.parse_args()


def set_api_key(key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path


def main(src_file, src_lang, dst_file, dst_lang, api_key_path):
    if api_key_path:
        set_api_key(api_key_path)
    print('Translating {} from {} to {}'.format(src_file, src_lang, dst_lang))
    translator = GoogleTranslator(src_lang, dst_lang)
    subs = pysrt.open(src_file)
    for sub in subs:
        if sub.text:
            sub.text = translator.translate(sub.text)
    subs.save(dst_file, encoding='utf-8')
    print('Wrote {} captions to {}'.format(dst_lang, dst_file))
