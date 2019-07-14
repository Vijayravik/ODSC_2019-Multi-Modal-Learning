import bz2
import os
from urllib.request import urlopen
from face_recognition_model import create_model
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
from data import triplet_generator
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import warnings
# Suppress LabelEncoder warning
from sklearn.manifold import TSNE

import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import glob
import IPython
#from td_utils import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
import matplotlib.mlab as mlab