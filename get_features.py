
import os
import numpy as np
import pickle

from get_mobilenet import get_mobilenet_features
from Get_class import encode, decode

def get_features(data_path):
    imgl = []
    for each in os.listdir(data_path):
        each = os.path.join(data_path,each)
        for e in os.listdir(each):
            img_path = os.path.join(each,e)
            imgl.append(img_path)
    x,y = get_x_y_array(imgl)
    return np.asarray(x), np.asarray(y)

def get_x_y_array(pathl):
    x = get_mobilenet_features(pathl,360)
    y = encode(pathl,True)
    return x,y