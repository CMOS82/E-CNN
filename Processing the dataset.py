#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Preparing the enhanced  images
import cv2 
import os

import numpy as np

import skimage as sk
from skimage import transform

from random import seed
from random import randint

from matplotlib import pyplot as plt


#Import required image modules
from PIL import Image, ImageFilter

#Import all the enhancement filter from pillow
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)

size = 227

def cropping2(img):
    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)

    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    center_coordinates = (int(img.shape[0]/2), int(img.shape[1]/2))

    radius = int(img.shape[1]*0.9) 

    color = (255, 255,255)

    thickness = -1

    mask = cv2.circle(mask,center_coordinates , radius, color, thickness)

    output = np.where(mask==np.array([255, 255, 255]), img, blurred_img)

    #remove unwanted spaces black edge bu checking before if the resulted image from the grabcut is none
    mask = np.zeros(output.shape[:2],np.uint8)
    output = np.where(mask==np.array([255, 255, 255]), output, output)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    imageHeight = output.shape[:2][0]
    imageWidth = output.shape[:2][1]
    rect = (1,1, imageHeight , imageWidth)
    cv2.grabCut(output,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    output = output*mask2[:,:,np.newaxis]
    return output

def cropping(img):
    
    #####
    #Original Grabcut
    #####
    mask_grab = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    # Define boundary rectangle containing the foreground object
    height, width, _ = img.shape
    left_margin_proportion = 0.3
    right_margin_proportion = 0.3
    up_margin_proportion = 0.1
    down_margin_proportion = 0.1

    boundary_rectangle = (
    int(width * left_margin_proportion),
    int(height * up_margin_proportion),
    int(width * (1 - right_margin_proportion)),
    int(height * (1 - down_margin_proportion)),
    )

    #boundary_rectangle = (50,50,590,376)
    cv2.grabCut(img,mask_grab,boundary_rectangle,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask_grab==2)|(mask_grab==0),0,1).astype('uint8')
    img_grab = img*mask2[:,:,np.newaxis]
    #cv2.imwrite('D:/Deep Learning/datasets/test/Grab.jpg', img_grab)

    #######

    ###My proposed method
    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)

    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    
    
    x = img.shape[1]
    y = img.shape[0]
    
    center_coordinates = (int(y * 0.5 ), int(x * 0.53))
    
    if(x>y):
        major = int( 0.75 *x)
        minor = int(0.6 * y)
    
    else:
        major = int( 0.6 *y)
        minor = int(0.75 * x)
        
    axesLength = (major, minor)
    angle = 0

    startAngle = 0

    endAngle = 360
   
    
    color = (255, 255,255)

    thickness = -1

    #mask = cv2.circle(mask,center_coordinates , radius, color, thickness)
    mask= cv2.ellipse(mask, center_coordinates, axesLength,
           angle, startAngle, endAngle, color, thickness)

    output = np.where(mask==np.array([255, 255, 255]), img, blurred_img)

    #remove unwanted spaces black edge but checking before if the resulted image from the grabcut is none
    mask = np.zeros(output.shape[:2],np.uint8)
    output = np.where(mask==np.array([255, 255, 255]), output, output)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    imageHeight = output.shape[:2][0]
    imageWidth = output.shape[:2][1]
    rect = (1,1, imageHeight , imageWidth)
    cv2.grabCut(output,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    output = output*mask2[:,:,np.newaxis]
    
    
    
    gray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
            
    w= gray.shape[1]
    h= gray.shape[0]

    TotalNumberOfPixels = w * h;
    WhitePixels = cv2.countNonZero(gray)
    BlackPixels = TotalNumberOfPixels - WhitePixels


    if( BlackPixels == TotalNumberOfPixels |  BlackPixels >=0.5 * TotalNumberOfPixels ):
        output0 = img
    
    elif( WhitePixels == TotalNumberOfPixels |  WhitePixels >=0.5 * TotalNumberOfPixels ):
        output0 = img
    else:
        output0 = output
        
        
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

        contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        cnt = contours[0]
        c = max(contours, key= cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        img0 = output0[y:y+h,x:x+w]
        output0 = img0
        
        cv2.imwrite('D:/Deep Learning/datasets/test/output2.jpg', output0)
        
        
    return output0

def add_gaussian_noise(img):
    
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (size, size)) #  np.zeros((224, 224), np.float32)
    
    img= image_centering(img, size, 0)


    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image

def image_centering(src_image, size, gray):
            
            s_img = src_image
            #l_img = np.zeros((size, size, 1), dtype = "uint8")
            #Reading a non existing image for black background
            if(gray):
                l_img = cv2.imread("C:/Users/engsh/Documents/Generating Sketch dataset/black_background.jpg", cv2.IMREAD_GRAYSCALE)
            else:
                l_img = cv2.imread("C:/Users/engsh/Documents/Generating Sketch dataset/black_background.jpg")
            l_img=cv2.resize(l_img,(size,size))
            

            width= s_img.shape[1] 
            height= s_img.shape[0] 

            w=0
            h = 0
            if(height>width):
                scale = size/height
                w=1
            elif(width>height):
                scale = size/width
                h=1
            else:
                scale = size/width



            width2 = int(width * scale)
            height2 = int(height * scale)

            s_img2 = cv2.resize(s_img,(width2,height2))

            if(w):
                start_corner = int((size - width2)/2)
                center =int(l_img.shape[1])
                l_img[0:height2, start_corner:start_corner+width2] = s_img2
            elif(h):
                start_corner = int((size - height2)/2)
                center =int(l_img.shape[0])
                l_img[start_corner:start_corner+height2, 0:width2] = s_img2
            else:
                l_img[0:height2, 0:width2] = s_img2
                
            return l_img
        
def sketch_image(src_img):
    s_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
            
    #Reducing the pixels of images, by having a dotted lines instead of straight lines

    img_invert = cv2.bitwise_not(s_img)

    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)


    sketch_img = cv2.divide(s_img, 255 - img_smoothing, scale=255)

    sketch_img = 255-sketch_img

    #Call the function of centering image with resizing
    sketch_img =image_centering(sketch_img, size, 1)
    
    return sketch_img


#Output image folder canny

sketch_train_folder = 'D:/Deep Learning/datasets/PASCAL/original70-30/train' 
sketch_test_folder = 'D:/Deep Learning/datasets/PASCAL/original70-30/test' 


# Input images folder name
training_img_folder= 'D:/Deep Learning/datasets/PASCAL/original00'



#Parent Directory
parent_dir ='C:/Users/engsh/Gesture Commands2'

# Chossing the augmentaion type
aug = 0

# the strart degree of rotation
degree = -1

#For counting the black pixels
count =0

for sub_folder_name in os.listdir(training_img_folder):
    path = os.path.join(training_img_folder, sub_folder_name)
    
    #Create a subdirectory with the sub-folder_name
   
    # Path2 for canny images
    path2 = os.path.join(sketch_train_folder, sub_folder_name)

    #Get the label of the sub_folder_name from the subfolders inside the training folder
    os.mkdir(path2)
    counter=0
    total= len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))])
    print(total)
    
    #Divide the images into 70 to 30 %
    percentage= 0.7
    
    for fileName in os.listdir(path):
        if (fileName.endswith(".JPEG")|fileName.endswith(".jpeg")|fileName.endswith(".jpg")):
        #if fileName.endswith(".JPEG"):
        #if fileName.endswith(".jpeg"):
            #img = cv2.imread(os.path.join(path, fileName), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(os.path.join(path, fileName))
            
            #Check whether the image is valid and read correctly
            if img is not None:
                
                 # Change the current directory to specified output directory 
                counter= counter + 1
                if(counter== round(total * percentage)):

                    path2 = os.path.join(sketch_test_folder, sub_folder_name)
                    os.mkdir(path2)
                    os.chdir(path2)
         
                if (aug == 0):
                    # seed random number generator
                    #seed(1)
                    #degree = randint(-25, 25)
                    #img= 255 - img

                    transformed_image = sketch_image(img)
                    transformed_image = 255 - transformed_image

                    # dividing height and width by 2 to get the center of the image
                    height, width = transformed_image.shape[:2]

                    # get the center coordinates of the image to create the 2D rotation matrix

                    center = (width/2, height/2)

                    # using cv2.getRotationMatrix2D() to get the rotation matrix

                    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=degree, scale=1)

                    # rotate the image using cv2.warpAffine

                    transformed_image = cv2.warpAffine(src=transformed_image, M=rotate_matrix, dsize=(width, height))
                    
                    transformed_image = cv2.resize(transformed_image, (size, size), fx=1.5, fy=1.5)
                    transformed_image = 255 - transformed_image

                  
                    aug = 1

                    degree +=1

                    if(degree == 1):
                        degree= -1
                
                elif(aug == 1):
                
                
                    #Flipp Horizontal
                   

                    transformed_image = sketch_image(img)
                    transformed_image = 255 - transformed_image


                    transformed_image = cv2.flip(transformed_image, 1)
                   
                    transformed_image = cv2.resize(transformed_image, (size, size), fx=1.5, fy=1.5)
                    
                    transformed_image = 255 - transformed_image


                    aug = 2
                   
               
                else:
                   
                    
                    transformed_image = add_gaussian_noise(img)
                                       
                    #upscale image
                    transformed_image = cv2.resize(transformed_image, (size, size), fx=2, fy=2)
                   
                    transformed_image = 255 - transformed_image
                    
                    aug=0

                
              
                # Change the current directory to specified output directory 
                os.chdir(path2)


                #cv.imwrite(fileName, canny, [int(cv.IMWRITE_JPEG_QUALITY), 100])
                #cv2.imwrite(fileName, canny_m)
                cv2.imwrite(fileName, transformed_image)

    #Return to the original training path
    os.chdir(path)

print("Finished Creating images")