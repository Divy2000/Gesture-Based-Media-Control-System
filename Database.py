
from numpy import asarray
from PIL import Image
from tensorflow import keras
import skimage.io as io
import numpy as np
from tensorflow.keras.preprocessing import image

def save(path, data):
    if ".npy" in path:
        np.save(path,data)
    elif str(type(data)) == "<class 'numpy.ndarray'>":
        save_image(path, data)
    else :
        save_model(path, data)      
    return path
        

def fetch(path, size = None):
    if size is None:
        if ".sav" in path and "model" in path:
            load_model(path)
            return load_model(path)
        elif ".npy" in path:
            return np.load(path)
        else:
            load_image(path)
            return load_image(path)
    elif isinstance(size,tuple):
        img = image.load_img(path, target_size=size)
        img = image.img_to_array(img)
        return img
    else:
        print("Please enter tuple in size")

def load_image(path):
    image = Image.open(path)
    image = asarray(image)
    return image

            
def save_image(path, image):
    io.imsave(path, image, check_contrast=False) 
    return path

        
def load_model(path):
    # from xgboost import XGBClassifier
    # from xgboost import Booster
    # # model = Booster()
    # # model = model.load_model(path)
    # clf = XGBClassifier()
    # booster = Booster()
    # booster.load_model(path)
    # clf._Booster = booster
    # # print("miracle")
    import pickle
    loaded_model = pickle.load(open(path, 'rb'))
    # print("new load used")
    return loaded_model    


def save_model(path, model):
    # model.save_model(path)
    import pickle
    # filename = 'finalized_model.sav'
    pickle.dump(model, open(path, 'wb'))
    # print("new save used")
    return path