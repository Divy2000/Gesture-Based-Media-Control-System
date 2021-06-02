
from sklearn.model_selection import train_test_split
from Get_class import decode, encode

import os
import pickle
import numpy as np

def normalize(x):
    xmax, xmin = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    return x

def get_label_code():
    dict_p = os.path.join("dict","dict_e.pkl")
    a_file = open(dict_p, "rb")
    output = pickle.load(a_file)
    output = list(output.keys())
    return encode(output,False)

def class_balance(x,y):
    from random import randrange
    s = y.shape[1]
    num = np.zeros(s)
    for each in y:
        i=0
        for e in each:
            if e == 1:
                num[i] = num[i]+1
            i=i+1
    min_=min(num)
    i=0
    for e in num:
        if e == min_:
            num = np.zeros(s).tolist()
            num[i] = 1
            num = np.asarray(num)
            print(f"{decode([num])[0]} is having minimum data of {min_}")
            break
        i=i+1
    num = np.zeros(s)
    came_index = []
    x_b=[]
    y_b=[]
    classes = get_label_code()
    # boolean = True
    while min(num) != min_ or max(num) != min_:
        j=0
        while j<len(num):
            if num[j]<min_:
                index = randrange(y.shape[0])
                # print(num)
                if index not in came_index:
                    if (y[index] == classes[j]).all():
                        x_b.append(x[index])
                        y_b.append(y[index])
                        came_index.append(index)    
                        num[j] = num[j] + 1
            j=j+1
    x_b = np.asarray(x_b)
    y_b = np.asarray(y_b)
    return x_b,y_b

def tts(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2)
    return normalize(x_train), normalize(x_test), normalize(x_val), normalize(y_train), normalize(y_test), normalize(y_val)