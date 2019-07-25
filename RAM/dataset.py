import random
import numpy as np
import cv2 as cv
import chainer

from pathlib import Path
from sklearn.model_selection import train_test_split
from chainer import cuda

xp = cuda.cupy
cuda.get_device(0)


class DatasetLoader:
    def __init__(self, path: Path):
        self.path = path
        self.pathlist = list(self.path.glob('**/*.jpg'))
        self.train, self.valid = self._split(self.pathlist)
        self.train_len = len(self.train)
        self.valid_len = len(self.valid)

    def __str__(self):
        return f"dataset path: {self.path} train data: {self.train_len}"

    def _split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.9)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    @staticmethod
    def _random_crop(image, size=256):
        height, width = image.shape[0], image.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        cropped = image[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return cropped

    @staticmethod
    def _coordinate(image):
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _variable(image_list):
        return chainer.as_variable(xp.array(image_list).astype(xp.float32))

    def _prepare_pair(self, image_path, crop_size=256):
        image = cv.imread(str(image_path))

        interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )

        interpolation = random.choice(interpolations)

        hr_image = self._random_crop(image, size=crop_size)
        down_size = int(crop_size / 4)
        sr_image = cv.resize(hr_image, (down_size, down_size), interpolation=interpolation)

        hr_image = self._coordinate(hr_image)
        sr_image = self._coordinate(sr_image)

        return (hr_image, sr_image)

    def __call__(self, batchsize, mode='train', crop_size=256):
        hr_box = []
        sr_box = []
        for _ in range(batchsize):
            if mode == 'train':
                rnd = np.random.randint(self.train_len)
                image_path = self.train[rnd]
            elif mode == 'valid':
                rnd = np.random.randint(self.valid_len)
                image_path = self.valid[rnd]
            else:
                raise AttributeError

            hr, sr = self._prepare_pair(image_path, crop_size=crop_size)

            hr_box.append(hr)
            sr_box.append(sr)

        return self._variable(hr_box), self._variable(sr_box)
