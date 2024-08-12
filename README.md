# Hand Gesture Recognition for Human-Computer Interaction

This project demonstrates a gesture-based media control system that allows users to control media playback with hand gestures. It includes functionalities for play/pause, volume up, and volume down through a gesture recognition model integrated with media controls.

## Project Highlights

- **Dataset**: Curated a dataset of hand gestures for media control functions (play/pause, volume up, volume down).
- **Preprocessing**: Applied preprocessing and augmentation techniques to enhance gesture recognition accuracy to 95%.
- **Model Development**: Developed a gesture recognition model and integrated it with media controls for hands-free operation.

## Demo Link 
[Demo of Hand Gesture Recognition For Human Computer Interaction](https://youtu.be/TA2x1n3wu9c)

## Folder Structure
```
📦AI model
 ┣ 📂datasets
 ┃ ┣ 📂Down
 ┃ ┣ 📂Ideal
 ┃ ┣ 📂Stop
 ┃ ┗ 📂Up
 ┣ 📂dict
 ┃ ┣ 📜dict_d.pkl
 ┃ ┣ 📜dict_e.pkl
 ┃ ┣ 📜dict_ld.pkl
 ┃ ┗ 📜dict_le.pkl
 ┣ 📂model
 ┃ ┣ 📂assets
 ┃ ┣ 📂variables
 ┃ ┣ 📜keras_metadata.pb
 ┃ ┗ 📜saved_model.pb
 ┣ 📂numpy_data
 ┃ ┣ 📜x_test.npy
 ┃ ┣ 📜x_train.npy
 ┃ ┣ 📜x_val.npy
 ┃ ┣ 📜y_test.npy
 ┃ ┣ 📜y_train.npy
 ┃ ┗ 📜y_val.npy
 ┣ 📜camera_pred.py
 ┣ 📜Database.py
 ┣ 📜driver.py
 ┣ 📜gesture.py
 ┣ 📜Get_class.py
 ┣ 📜get_features.py
 ┣ 📜get_mobilenet.py
 ┣ 📜model.py
 ┗ 📜train_test_split.py
```
