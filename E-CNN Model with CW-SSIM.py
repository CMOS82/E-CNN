#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#### 1 ###
#E-CNN Model
import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D, GaussianNoise, BatchNormalization
from keras.models import Sequential
import tensorflow as tf
import os
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
import time
from keras.layers import MaxPooling2D


earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)

callbacks = [earlystop,learning_rate_reduction]


batch = 32
size =227
input_shape = (size,size,3)


# Initilizes the model parameters
def def_model_param():
    GESTURE_CATEGORIES = len(CATEGORY_MAP)
    base_model = Sequential()
    base_model.add(SqueezeNet(input_shape=input_shape, include_top=False))

    base_model.add(MaxPool2D())
    
    #my addition
  
    base_model.add(BatchNormalization())
   
    base_model.add(Dropout(0.5))
   
    
    base_model.add(Convolution2D(GESTURE_CATEGORIES, (3, 3), padding='valid'))
   
   
    base_model.add(Dropout(0.2))
    #
    
    base_model.add(GlobalAveragePooling2D())
   
   
    base_model.add(Activation('softmax'))
   
    return base_model

# This function returns the numeric equivalent of the category/class name    
def label_mapper(val):
    return CATEGORY_MAP[val]

# Input images folder name

training_img_folder ='D:/Deep Learning/datasets/PASCAL/sketch/train3'
test_img_folder = 'D:/Deep Learning/datasets/PASCAL/sketch/test3' 




CATEGORY_MAP = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
    
}


# Loading the input t images from all the folders into 'input_data' variable
input_data = []

for sub_folder_name in os.listdir(training_img_folder):
    path = os.path.join(training_img_folder, sub_folder_name)
    count1=0
    for fileName in os.listdir(path):
       
        if (fileName.endswith(".jpeg")|fileName.endswith(".JPEG")|fileName.endswith(".jpg")):
       
            img = cv2.imread(os.path.join(path, fileName))
            img = cv2.resize(img, (size, size))
            
            #'input_data' stores the input image array and its corresponding label or category name
            input_data.append([img, sub_folder_name])
           
# Zip function to separate the 'img_data'(input image) & 'labels' (output text labels) 
img_data, labels = zip(*input_data)

# Converting text labels to numeric value as per CATEGORY_MAP
# Eg:- ["up","up",down","mute",..] -> [0,0,1,2,..]
# Python 'map' function takes 2 arguments:
# 1. A function (label_mapper)
# 2. An iterable 
labels = list(map(label_mapper, labels))


# Converting numeric labels and performing one hot encoding on them
# Eg:- [0,0,1,2] -> [[1 0 0 0 0 0]
#                    [1 0 0 0 0 0]
#                    [0 1 0 0 0 0]
#                    [0 0 1 0 0 0]]
labels = np_utils.to_categorical(labels)


#preparing the validation data

# Loading the input validating images from all the folders into 'test' variable
test_data = []
#count2=0
for sub_folder_name in os.listdir(test_img_folder):
    path = os.path.join(test_img_folder, sub_folder_name)
    
    for fileName in os.listdir(path):
        
        if (fileName.endswith(".jpeg")|fileName.endswith(".JPEG")|fileName.endswith(".jpg")):
            img = cv2.imread(os.path.join(path, fileName))
            img = cv2.resize(img, (size, size))

            #'input_data' stores the input image array and its corresponding label or category name
            test_data.append([img, sub_folder_name])
            

# Zip function to separate the 'img_data'(input image) & 'labels' (output text labels) 
img_data_test, labels2 = zip(*test_data)

# Converting text labels to numeric value as per CATEGORY_MAP
# Eg:- ["up","up",down","mute",..] -> [0,0,1,2,..]
# Python 'map' function takes 2 arguments:
# 1. A function (label_mapper)
# 2. An iterable 
labels2 = list(map(label_mapper, labels2))


# Converting numeric labels and performing one hot encoding on them
# Eg:- [0,0,1,2] -> [[1 0 0 0 0 0]
#                    [1 0 0 0 0 0]
#                    [0 1 0 0 0 0]
#                    [0 0 1 0 0 0]]
labels2 = np_utils.to_categorical(labels2)



# define the model
model = def_model_param()
model.compile(
    
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['acc']
)

print(model.summary())



# fit mo
history= model.fit(np.array(img_data), np.array(labels), batch_size= 32 , epochs=5, 
                   validation_data= [np.array(img_data_test), np.array(labels2)], callbacks= callbacks)               
           

print("Training Completed")

# save the trained model parameters into a .h5 file
model.save("my_train.h5")


# In[2]:
### 2 ###
###Plotting

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc,  "mo--", linewidth = 3,
        markersize = 10, label = "P. Training Accuracy")
plt.plot(epochs, val_acc, "yo:", linewidth = 3,
        markersize = 10, label = "P. Validation Accuracy")
plt.title('Training and validation accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy(%)")
plt.legend(loc = 4)
plt.grid()
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, "bx-", linewidth = 3,
        markersize = 10, label ="Training Loss")
plt.plot(epochs, val_loss, "kx-", linewidth = 3,
        markersize = 10, label ="Validation Loss")
plt.title('Training and validation loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc = 0)
plt.grid()
plt.figure()

###

#######
### 3 ###
#Calcualting the similarity factor; in order to determine the optimum batch size
%matplotlib inline

import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

from ssim import SSIM
from ssim.utils import get_gaussian_kernel

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
# Input images folder name
training_img_folder = 'D:/Deep Learning/datasets/animals10/sketch227/train'

# Loading the input training images from all the folders into 'input_data' variable
input_data = []
similarity_list = []
ssim = 0
cw_ssim = 0
tot_avg = 0 

#size = (225,225)
size = (8,8)


counter = 1
Total_samples = 0
average_number =0


#Iterate through the subfolders
for sub_folder_name in os.listdir(training_img_folder):
    path = os.path.join(training_img_folder, sub_folder_name)
    for fileName in os.listdir(path):
        if fileName.endswith(".jpeg"):
            
            #reading a numpy file image
            im = Image.open(os.path.join(path, fileName))
            # Resizing the image to avoid aliasing
            #im = im.resize(size, Image.ANTIALIAS)
            #Convert the colorful image to grayscale
            im_gray = ImageOps.grayscale(im)
            
            Total_samples +=1
            
             #'input_data' stores the input image array and its corresponding label or category name
            #input_data.append([img, sub_folder_name])
            
            if counter == 1:
                im1=im_gray
                counter += 1
            else:
                #calculate the SSIM index
                ssim = SSIM(im1, gaussian_kernel_1d).ssim_value(im_gray)
                
                #calculate the CW-SSIM
                cw_ssim = SSIM(im1).cw_ssim_value(im_gray)

                
    #Calculate the average similarity index per each subfolder
    avg_sim = (cw_ssim * 0.7 + ssim * 0.3)
    
    #calculate the number of averages 
    average_number +=1
    print("average similarity of input image %.4f" % avg_sim)
    
    
    if avg_sim <2:
        Batch_Size = 0      
    elif avg_sim <4:
        Batch_Size = 2
    elif avg_sim <8:
        Batch_Size = 4
    elif avg_sim <16:
        Batch_Size = 8
    elif avg_sim <32:
        Batch_Size = 16
    elif avg_sim <64:
        Batch_Size = 32
    elif avg_sim >64:
        Batch_Size = 64
        
        
    #calculating the total average of the whole traing images
    tot_avg += avg_sim
    
Average_total_sim = (tot_avg / average_number)  

Average_total_sim = Average_total_sim*100


print("Recommended Batch Size = %.4f" % Average_total_sim)         









