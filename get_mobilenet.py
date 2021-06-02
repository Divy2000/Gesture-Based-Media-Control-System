

from Database import save,fetch

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
from pathlib import Path
import os

model = MobileNetV2(input_shape=(360,360,3),weights="imagenet", include_top=False)

def get_mobilenet_features(path_list, s):
    folder_path = None
    feature = []
    came = []
    # i=1
    for e in path_list:
        # print(i)
        # i=i+1
        folder_path_temp = str(Path(e).parent.absolute())
        # if folder_path != folder_path_temp:
        #     folder_path = folder_path_temp
        #     i=1
        x = fetch(e,(s,s))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = model.predict(x)
        x = x[0]
        feature.append(x)
    feature = np.asarray(feature)
    return feature