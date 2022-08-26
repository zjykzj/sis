# -*- coding:utf-8-*-

import os
import logging

import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template

from feature_extractor import FeatureExtractor

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

IMG_ROOT = 'static/img'
IMG_SUFFIX = '.jpg'

FEATURE_ROOT = 'static/feature'
FEATURE_SUFFIX = '.npy'


def init():
    features = []
    img_paths = []
    for item in Path(FEATURE_ROOT).rglob(f"*{FEATURE_SUFFIX}"):
        feature_path = str(item)

        img_path = feature_path.replace(FEATURE_ROOT, IMG_ROOT).replace(FEATURE_SUFFIX, IMG_SUFFIX)
        if not os.path.isfile(img_path):
            continue

        features.append(np.load(feature_path))
        img_paths.append(img_path)

    app.logger.info(f"feature len {len(features)}")

    features = np.array(features)
    return features, img_paths


fe = FeatureExtractor()
features, img_paths = init()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        # L2 distances to features
        app.logger.info(f"features{features.shape} vs. query{query.shape}")
        dists = np.linalg.norm(features - query, axis=1)
        # Top 30 results
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0")
