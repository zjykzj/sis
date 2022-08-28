# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from feature_extractor import FeatureExtractor

IMG_SUFFIX = '.jpg'
FEATURE_SUFFIX = '.npy'


def save_feature(src_img_path, src_root, dst_root, feature):
    dst_feature_path = src_img_path.replace(src_root, dst_root).replace(IMG_SUFFIX, FEATURE_SUFFIX)

    dst_feature_dir = os.path.dirname(dst_feature_path)
    if not os.path.exists(dst_feature_dir):
        os.makedirs(dst_feature_dir)

    np.save(dst_feature_path, feature)


def main():
    fe = FeatureExtractor()
    print(fe)

    src_root = "static/img"
    dst_root = 'static/feature'

    for item in tqdm(list(Path(src_root).rglob('*.jpg'))):
        img_path = str(item)
        img = Image.open(img_path)

        feature = fe.extract(img)
        if len(feature.shape) != 1:
            feature = feature.reshape(-1)

        save_feature(img_path, src_root, dst_root, feature)


if __name__ == '__main__':
    main()
