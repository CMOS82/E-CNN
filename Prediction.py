#!/usr/bin/env python
# coding: utf-8

###Prediction 

from keras.models import load_model
import cv2
import numpy as np
import sys
import time

size = (227,227)
# starting time
start = time.time()

filepath = sys.argv[1]

CATEGORY_MAP = {
    0 : "airplane",
    1 : "car",
    2 : "cat",
    3 : "dog",
    4 : "flower",
    5 : "fruit",
    6 : "motorbike",
    7 : "person"
    
}



# This function returns the gesture name from its numeric equivalent 
def mapper(val):
    return CATEGORY_MAP[val]

#Load the saved model file
model = load_model("my_train.h5")

# Ensuring the input image has same dimensions that is used during training. 
img = cv2.imread('D:/Deep Learning/datasets/natural_images/original/test/cat/cat_0842.jpg')

img = cv2.resize(img, size)



# Predict the gesture from the input image
prediction = model.predict(np.array([img]))

gesture_numeric = np.argmax(prediction[0])
gesture_name = mapper(gesture_numeric)

# sleeping for 1 sec to get 10 sec runtime
#time.sleep(1)

# end time
end = time.time()

print("Predicted Gesture: {}".format(gesture_name))

# total time taken
print(f"Runtime of the prediction is {end - start}")
