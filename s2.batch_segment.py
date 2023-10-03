"""
Using https://github.com/facebookresearch/segment-anything

prepare the environment
prepare a recorded_points file
prepare the images


"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from os.path import *
from segment_anything import SamPredictor, sam_model_registry
checkpoint = "/mnt/ivy/thliao/software/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)


def remove_background(image, bg_color=255):
    # assumes rgb image (w, h, c)
    intensity_img = np.mean(image, axis=2)

    # identify indices of non-background rows and columns, then look for min/max indices
    non_bg_rows = np.nonzero(np.mean(intensity_img, axis=1) != bg_color)
    non_bg_cols = np.nonzero(np.mean(intensity_img, axis=0) != bg_color)
    r1, r2 = np.min(non_bg_rows), np.max(non_bg_rows)
    c1, c2 = np.min(non_bg_cols), np.max(non_bg_cols)

    # return cropped image
    return image[r1:r2+1, c1:c2+1, :]

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def process(img_path,points,predictor,name,odir='./'):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    input_point = np.array([*points])
    input_label = np.array([1]*len(points))

    masks, scores, logits = predictor.predict(point_coords=input_point,
                                              point_labels=input_label,
                                              multimask_output=True,)
    if len(points)==1:
        mask = masks[np.argmax(scores)]
    else:
        mask_input = logits[np.argmax(scores), :, :]
        masks, _, _ = predictor.predict(point_coords=input_point,
                                        point_labels=input_label,
                                        mask_input=mask_input[None, :, :],
                                        multimask_output=False,)
        mask = masks[0]
    i_copy = image.copy()
    i_copy[mask==False] = [255,255,255]
    i_copy = remove_background(i_copy)
    d = dirname(f'{odir}/{name}')
    if not exists(d):
        os.system(f"mkdir -p {d}")
    t = cv2.cvtColor(i_copy, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{odir}/{name}',t)

import os
from glob import glob
from tqdm import tqdm
import cv2
from collections import defaultdict

if __name__ == '__main__':
    output_file = '../recorded.txt'
    indir = './'
    recorded_f = [_.split('\t')
                  for _ in open(output_file).read().strip().split('\n')]
    name2points = {k[0].replace('\\','/'):[(int(e.split(',')[0]),int(e.split(',')[1]))
                        for e in k[1:]]
                  for k in recorded_f}
    all_f = glob(f'{indir}/**/*.JPG')
    for f in tqdm(all_f):
        name = f.replace(indir,'',1)
        if name not in name2points:continue
        points = name2points[name]
        predictor = SamPredictor(sam)
        process(f,points,predictor,name,odir='./found/')
