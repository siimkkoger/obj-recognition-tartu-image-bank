import matplotlib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import os
import copy

# Some setup first as usual
from fastai.vision import *
from fastai.metrics import error_rate

# We don't like warnings
import warnings
warnings.filterwarnings('ignore')

from urllib.request import urlopen

np.random.seed(1)

EPSILON = 1e-5
# Load GTSDB validation images
CLASSES = ['prohibitory', 'mandatory', 'danger']
# Path to the German Traffic Sign Detection Benchmark dataset
DATASET_PATH = '~/Datasets/GermanTSDataset/Detection'
full_set = ['%05d.jpg' % x for x in range(900)]
valid_set = full_set[600:900]

MODEL_NAME = 'faster_rcnn_resnet101'

#l = None
#with open('detections_output_result.pkl', 'rb') as pickle_file:
#    l = pickle.load(pickle_file)
l = load_learner('learners/l1')

i = 149
test_img = open_image(f'images/img{i}.jpg', convert_mode='RGB')
pred_class, pred_idx, outputs = l.predict(test_img)
print(pred_class)
