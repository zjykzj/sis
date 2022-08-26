# -*- coding: utf-8 -*-

"""
@date: 2022/8/26 下午5:39
@file: search.py
@author: zj
@description: 
"""

import os
import logging

from PIL import Image
import numpy as np
from pathlib import Path

from feature_extractor import FeatureExtractor

IMG_ROOT = 'static/img'
IMG_SUFFIX = '.jpg'

FEATURE_ROOT = 'static/feature'
FEATURE_SUFFIX = '.npy'

logger = logging.getLogger(__name__)


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

    logger.info(f"Feature len {len(features)}")

    features = np.array(features)
    return features, img_paths


class Search:

    def __init__(self):
        self.fe = FeatureExtractor()
        self.features, self.img_paths = init()

    def run(self, img: Image):
        # Run search
        query = self.fe.extract(img)
        # L2 distances to features
        logger.info(f"Features{self.features.shape} vs. Query{query.shape}")
        dists = np.linalg.norm(self.features - query, axis=1)
        # Top 30 results
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], self.img_paths[id]) for id in ids]

        return scores
