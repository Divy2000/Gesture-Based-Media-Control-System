

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle

from pathlib import Path
from Database import save,fetch
import os

path_d = "dict"
dic_e_g = None
dic_d_g = None
dic_le_g = None
dic_ld_g = None
dict_e_p = os.path.join(path_d,"dict_e.pkl")
dict_d_p = os.path.join(path_d,"dict_d.pkl")
dict_le_p = os.path.join(path_d,"dict_le.pkl")
dict_ld_p = os.path.join(path_d,"dict_ld.pkl")

def save_dict(dict_p,dic):
    a_file = open(dict_p, "wb")
    pickle.dump(dic, a_file)
    a_file.close()
    return None

def fetch_dict(dict_p):
    a_file = open(dict_p, "rb")
    output = pickle.load(a_file)
    return output

def encode_in(y,dic_e):
    y = [dic_e[e] for e in y]
    return y

def decode_in(y,dic_d):
    y = [dic_d[str(e)] for e in y]
    return y
    
def make_dict():
    e = os.listdir("datasets")
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    e_le = label_encoder.fit_transform(e)
    e_e = e_le.reshape(len(e_le), 1)
    e_e = onehot_encoder.fit_transform(e_e)
    dic_e = {}
    dic_d = {}
    dic_le = {}
    dic_ld = {}
    i = 0
    while i<len(e):
        dic_e[e[i]] = e_e[i]
        dic_d[str(e_e[i])] = e[i]
        dic_le[e[i]] = e_le[i]
        dic_ld[str(e_le[i])] = e[i]
        i=i+1
    return dic_e,dic_d, dic_le, dic_ld

def compare_dict():
    dic_e,dic_d, dic_le, dic_ld = make_dict()
    dirs = os.listdir()
    if path_d not in dirs:
        os.mkdir(path_d)
        save_dict(dict_e_p,dic_e)
        save_dict(dict_d_p,dic_d)
        save_dict(dict_le_p,dic_le)
        save_dict(dict_ld_p,dic_ld)
        return dic_e
    dic_d_o = fetch_dict(dict_d_p) 
    if dic_d != dic_d_o:
        n_p = "numpy_data"
        if n_p not in dirs:
            save_dict(dict_e_p,dic_e)
            save_dict(dict_d_p,dic_d)
            save_dict(dict_le_p,dic_le)
            save_dict(dict_ld_p,dic_ld)
            return dic_e
        for each in os.listdir(n_p):
            if "y_" in each:
                path = os.path.join(n_p,each)
                y = fetch(path)
                y = decode_in(y,dic_d_o)
                y = encode_in(y,dic_e)
                save(path,y)
        save_dict(dict_e_p,dic_e)
        save_dict(dict_d_p,dic_d)
        save_dict(dict_le_p,dic_le)
        save_dict(dict_ld_p,dic_ld)
    return dic_e

def get_class(path):
    p1 = Path(path).parent.absolute()
    p2 = Path(p1).parent.absolute()
    p1 = str(p1)
    p2 = str(p2)
    c = p1.replace(f"{p2}\\","")
    return c

def encode(y,boolean):
    dic_e = compare_dict()
    if boolean:
        y = [get_class(e) for e in y]
    y = encode_in(y, dic_e)
    return y

def decode(y):
    dic_d = fetch_dict(dict_d_p)
    y = decode_in(y, dic_d)
    return y

def lencode(y):
    dic_le = fetch_dict(dict_le_p)
    y = encode_in(y,dic_le)
    return y

def ldecode(y):
    dic_ld = fetch_dict(dict_ld_p)
    y = decode_in(y,dic_ld)
    return y

def get_num_classes():
    dic = fetch_dict(dict_e_p)
    return len(dic)