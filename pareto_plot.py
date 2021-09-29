'''
Created on April , 2021
@author:
'''

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np

import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
# import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt





from utils.pareto import pareto
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

pop_size = 20
n_generations = 20

current_dir = os.path.dirname(os.path.abspath(__file__))

pic_dir = os.path.join(current_dir, 'Figures')
# Log file path of EA in csv
ea_log_path = os.path.join(current_dir, 'EA_log')

scale = 100

def roundup(x, scale):
    return int(math.ceil(x / float(scale))) * scale

def rounddown(x, scale):
    return int(math.floor(x / float(scale))) * scale


pd.options.mode.chained_assignment = None  # default='warn'
mute_log_file_path = os.path.join(ea_log_path, 'mute_log_%s_%s.csv' % (pop_size, n_generations))
# ea_log_path + 'mute_log_%s_%s.csv' % (pop_size, n_generations)
mute_log_df = pd.read_csv(mute_log_file_path)
solutions_df = mute_log_df[['val_rmse', 'penalty']]
solutions_df['penalty'] = solutions_df['penalty'] * 1e4
print (solutions_df)



#############################à
data = solutions_df
col_a = 'val_rmse'
col_b = 'penalty'

sets = {}
archives = {}

fig = matplotlib.figure.Figure(figsize=(15, 15))
agg.FigureCanvasAgg(fig)


# print ("data", data)
# print ("columns", data.columns)
# print ("data.itertuples(False)", data.itertuples(False))
resolution = 1e-4

archives = pareto.eps_sort([data.itertuples(False)], [0, 1], [resolution] * 2)
sets = pd.DataFrame(data=archives.archive)
# print ("archives", archives)
# print ("sets", sets)

spacing_x = 0.1
spacing_y = 500

fig = matplotlib.figure.Figure(figsize=(5, 5))
agg.FigureCanvasAgg(fig)

ax = fig.add_subplot(1, 1, 1)
ax.scatter(data[col_a], data[col_b], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1)
ax.scatter(sets[col_a], sets[col_b], facecolor=(1.0, 1.0, 0.4), zorder=1, s=50)

for box in archives.boxes:
    ll = [box[0] * resolution, box[1] * resolution]

    # make a rectangle in the Y direction
    rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), 1.4 - ll[0], 1.4 - ll[1], lw=0,
                                        facecolor=(1.0, 0.8, 0.8), zorder=-10)
    ax.add_patch(rect)

    # make a rectangle in the X direction
    rect = matplotlib.patches.Rectangle((ll[0] + resolution, ll[1]), 1.4 - ll[0], 1.4 - ll[1], lw=0,
                                        facecolor=(1.0, 0.8, 0.8), zorder=-10)
    ax.add_patch(rect)
if resolution < 1e-3:
    spacing = 0.2
else:
    spacing = resolution
    while spacing < 0.2:
        spacing *= 2

ax.set_xticks(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x))
ax.set_yticks(np.arange(rounddown(min(data[col_b])-100,scale), roundup(max(data[col_b])+100,scale), spacing_y))

# if resolution > 0.001:
#     ax.hlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
#     ax.vlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
ax.set_xlim(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2)
ax.set_ylim(rounddown(min(data[col_b])-200, 100), roundup(max(data[col_b])+200,100))
ax.set_title("Epsilon resolution: {0:.2g}".format(resolution))
ax.set_xlabel(r'$f_1$')
ax.set_ylabel(r'$f_2$')

fig.savefig("example")

fig = matplotlib.figure.Figure(figsize=(5, 5))
agg.FigureCanvasAgg(fig)

ax = fig.add_subplot(1, 1, 1)
ax.scatter(data[col_a], data[col_b], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1)

ax.set_xticks(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x))
ax.set_yticks(np.arange(rounddown(min(data[col_b])-100,scale), roundup(max(data[col_b])+100,scale), spacing_y))
ax.set_xlim(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2)
ax.set_ylim(rounddown(min(data[col_b])-200, 100), roundup(max(data[col_b])+200,100))
ax.set_title("Unsorted Data")
ax.set_xlabel(r'$f_1$')
ax.set_ylabel(r'$f_2$')

fig.savefig("unsorted")
