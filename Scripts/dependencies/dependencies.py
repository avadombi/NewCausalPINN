import tensorflow as tf
import keras.backend as K
import keras
from keras.layers import Layer, Dense, Conv1D, Flatten, MaxPool1D, Dropout, Input, GlobalMaxPool1D
from keras import initializers, constraints, Model
from keras.constraints import NonNeg
from keras import callbacks
from keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl
import shap
import os
import shutil

from typing import Callable, Any
from sklearn.model_selection import KFold
from pylab import text

import matplotlib.dates as m_dates
from matplotlib.dates import DateFormatter

import hydroeval as he

RANDOM_STATE = 42
EPSILON = 1e-6
FONT_SIZE = 22
LINE_WIDTH = 1.5
WIDTH = 0.5
COLORS = {
    'black': '#000000',
    'blue': '#3275a1',
    'orange': '#e1802c',
    'green': '#3a923a',
    'red': '#c03d3d',
    'purple': '#9372b2',
    'gray': '#c4c4c4'
}

CONSTRAINT_TYPE = [None, 'CRC', 'SCRC1', 'SCRC2', 'SCRC3']
NUM_CNN_KERNEL_LAYERS = {
    'SCRC1': (2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
    'SCRC2': (1, 4, 5, 8, 9, 12, 13, 16, 17, 20),
    'SCRC3': (3, 4, 7, 8, 11, 12, 15, 16, 19, 20)
}
