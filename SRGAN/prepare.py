import cv2
import os
import numpy as np

def prepare_dataset(filename):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]
        rnd1 = np.random.randint(height-386)
        rnd2 = np.random.randint(width-386)

        image = image[rnd1:rnd1+384, rnd2:rnd2+384]

        hr_image = image
        sr_image = cv2.resize(hr_image, (96,96),interpolation=cv2.INTER_CUBIC)

        hr_image = hr_image[:,:,::-1]
        hr_image = hr_image.transpose(2,0,1)
        hr_image = (hr_image-127.5)/127.5

        sr_image = sr_image[:,:,::-1]
        sr_image = sr_image.transpose(2,0,1)
        sr_image = (sr_image-127.5)/127.5

        return hr_image, sr_image