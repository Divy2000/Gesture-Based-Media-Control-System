

import os
os.chdir("D:\\Drive\\MY Computer\\study\\Sem6\\AI LAB\\AI model")

from get_features import get_features
from train_test_split import tts
from model import model_maker,train,predict
from Get_class import decode
from Database import save,fetch
from tensorflow.keras.models import load_model

np_data = "numpy_data"
if np_data not in os.listdir():
    os.mkdir("numpy_data")
    
data_dir = "datasets"
x,y = get_features(data_dir)
x_train, x_test, x_val, y_train, y_test, y_val = tts(x,y)
save(os.path.join(np_data,"x_train.npy"),x_train)
save(os.path.join(np_data,"x_test.npy"),x_test)
save(os.path.join(np_data,"x_val.npy"),x_val)
save(os.path.join(np_data,"y_train.npy"),y_train)
save(os.path.join(np_data,"y_test.npy"),y_test)
save(os.path.join(np_data,"y_val.npy"),y_val)

# x_train = fetch(os.path.join(np_data,"x_train.npy"))
# x_test = fetch(os.path.join(np_data,"x_test.npy"))
# x_val = fetch(os.path.join(np_data,"x_val.npy"))
# y_train = fetch(os.path.join(np_data,"y_train.npy"))
# y_test = fetch(os.path.join(np_data,"y_test.npy"))
# y_val = fetch(os.path.join(np_data,"y_val.npy"))

model = model_maker(x_train)
model = train(model,[x_train,x_val],[y_train,y_val])
model.save("model")

# model_t = load_model("model")
pred = predict(model,x_val)
y_test_d = decode(y_val)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_d, pred)
acc = accuracy_score(y_test_d, pred)
print(cm)
print(acc*100)