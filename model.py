
import math
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from Get_class import get_num_classes,ldecode

def model_maker(x_train):
    n_class = get_num_classes()
    shape = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
    input = tf.keras.Input(shape=(shape,))   
    x = layers.Dense(100,activation="relu")(input)
    output = layers.Dense(n_class,use_bias=False, activation="softmax")(x)
    model = tf.keras.Model(input,output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, "categorical_crossentropy")
    return model

def train(model,x, y):
    x_train, x_valid = x
    y_train, y_valid = y
    batch_size, epochs, verbose = [math.floor(x_train.shape[0]*0.4),20,1]
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    model.fit(x_train, y_train, batch_size, epochs,validation_data=(x_valid, y_valid))
    return model

def predict(model, x):
    x_reshape = x.reshape(x.shape[0], -1)
    y_pred = model.predict(np.asarray(x_reshape))
    # y_pred = encode(y_pred, False)
    # y_pred = np.asarray(y_pred)
    y_pred = np.argmax(y_pred,axis=1)
    y_pred = ldecode(y_pred)
    return y_pred