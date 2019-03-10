import os
import re

from google.cloud import translate
from collections import defaultdict


FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_language_code(t):
    return t.split('-')[0]


class GoogleTranslator(object):

    def __init__(self, src, dst):
        self.src = get_language_code(src)
        self.dst = get_language_code(dst)
        self.translator = translate.Client()

    def translate(self, phrase: str) -> str:
        return self.translator.translate(
            phrase, source_language=self.src, target_language=self.dst
        )['translatedText']


def load_dictionary(src, dst):
    dict_re = re.compile(r'^(.+)\s+(.+)$')
    dictionary = defaultdict(set)

    dict_path = os.path.join(FILE_DIR, 'dictionaries', '{}-{}.txt'.format(
                             src, dst))
    if os.path.exists(dict_path):
        with open(dict_path) as f:
            for line in f:
                m = dict_re.match(line.strip())
                if m:
                    s = m.group(1)
                    d = m.group(2)
                    dictionary[s].add(d)
    else:
        raise NotImplementedError(
            'Missing dictionary: {} -> {}'.format(src, dst))

    idict_path = os.path.join(FILE_DIR, 'dictionaries',
                              '{}-{}.txt'.format(dst, src))
    if os.path.exists(idict_path):
        with open(idict_path) as f:
            for line in f:
                m = dict_re.match(line.strip())
                if m:
                    d = m.group(1)
                    s = m.group(2)
                    dictionary[s].add(d)
    else:
        raise NotImplementedError(
            'Missing dictionary: {} -> {}'.format(src, dst))

    assert len(dictionary) > 0
    return dictionary


class DictionaryTranslator(object):

    def __init__(self, src, dst):
        self.src = get_language_code(src)
        self.dst = get_language_code(dst)
        self.dictionary = load_dictionary(src, dst)

    def translate(self, word: str) -> str:
        return self.dictionary.get(word.lower(), set())
