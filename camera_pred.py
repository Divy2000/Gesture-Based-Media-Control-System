

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

mnv_model = MobileNetV2(input_shape=(360,360,3),weights="imagenet", include_top=False)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui
from model import predict
import time

# define a video capture object 
vid = cv2.VideoCapture(0)
model = load_model("model")

while(True):       
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    
    new_img = cv2.resize(frame, (360, 360))
    new_img = np.float32(new_img)
    new_img = np.expand_dims(new_img, axis=0)
    new_img = preprocess_input(new_img)
    new_img = mnv_model.predict(new_img)[0]
    pred = predict(model,np.asarray([new_img]))
    print(pred)
    
    
    predicted_gesture = pred[0]
    if(predicted_gesture == 'Up'):
        pyautogui.press('volumeup')
    elif(predicted_gesture == 'Down'):
        pyautogui.press('volumedown')
    elif(predicted_gesture == 'Stop'):
        pyautogui.press('playpause')
        time.sleep(1)
    elif(predicted_gesture == 'Ideal'):
        print("Ideal")
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()