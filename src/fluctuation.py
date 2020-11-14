import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

src_img = cv2.imread('./img/lize_00811_0.png')

np.random.seed(42)

mean = np.mean(src_img,axis=(0,1))
std = np.std(src_img, axis=(0,1))

proc_img = src_img
proc_shape = proc_img.shape


for i in tqdm(range(50)):

    bias_map = (np.ones(proc_shape)*(np.random.rand(3)-0.5)*80).astype(np.int)
    noise = (np.random.rand(proc_shape[0],proc_shape[1],proc_shape[2])-0.5)*15

    fluc_img = proc_img.astype(np.int) + bias_map

    fluc_img[fluc_img > 255] = 255
    fluc_img[fluc_img < 0] = 0

    fluc_img = fluc_img.astype(np.uint8)

    resize_ratio = (np.random.rand(1)-0.5)*0.1 + 1.0
    resize_shape = (proc_shape[:2] * resize_ratio).astype(np.int)
    # fluc_img = cv2.resize(fluc_img,tuple(resize_shape),cv2.INTER_CUBIC)

    filename = Path('img/augumentation')
    cv2.imwrite(str(filename.joinpath(f'{str(i).zfill(3)}.png')),fluc_img)

    parameter = {}
    parameter['bias_map'] = bias_map[0,0,:].tolist()
    parameter['resize_shape'] = resize_shape.tolist()

    with open(f'img/augumentation/{str(i).zfill(3)}.json', 'w') as f:
        json.dump(parameter, f, indent=4)