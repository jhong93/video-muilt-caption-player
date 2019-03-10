#!/usr/bin/env python3

import argparse
import os
from flask import Flask, send_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')
    parser.add_argument('aligned_subs_file')
    return parser.parse_args()


def main(video_file, aligned_subs_file):
    assert os.path.exists(video_file)
    assert os.path.exists(aligned_subs_file)

    app = Flask(__name__)

    @app.route('/')
    def root():
        return send_file('html/home.html', mimetype='text/html')

    @app.route('/subs')
    def subs():
        return send_file(aligned_subs_file, mimetype='application/json')

    @app.route('/video')
    def video():
        return send_file(video_file, mimetype='video/mp4', conditional=True)

    app.run(host='localhost', port=8080)


if __name__ == '__main__':
    main(**vars(get_args()))
