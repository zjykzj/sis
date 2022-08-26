# -*- coding: utf-8 -*-

import torchvision.transforms as transforms
from torchvision.models import resnet18

import numpy as np
from PIL import Image


class FeatureExtractor:
    def __init__(self):
        self.model = resnet18(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            # take a 224x224 img as an input
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, img: Image):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        # Make sure img is color
        img = img.convert('RGB')

        inputs = self.transform(img).unsqueeze(0)

        feature = self.model(inputs).detach().numpy()
        return feature / np.linalg.norm(feature)  # Normalize


if __name__ == '__main__':
    data_np = np.random.randn(224 * 32 * 3).reshape((224, 32, 3)) * 100
    data_pil = Image.fromarray(data_np.astype(np.uint8))

    m = FeatureExtractor()

    feature = m.extract(data_pil)
    print(feature.shape)
