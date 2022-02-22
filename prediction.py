# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:58:46 2022

@author: asrar
"""
import numpy as np
import matplotlib.pyplot as plt 
import keras

#to predict new images
path = "dataset/Sample_Images"

import os
from keras.preprocessing import image

for i in os.listdir(path) :
    img = image.load_img(path+ '//' + i, target_size = (64, 64))
    plt.imshow(img)
    plt.show()    
   
    #Converting to Pixel_Vector
    X = image.img_to_array(img)
    X = X/255
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    #Predicting_Results
    from keras.models import load_model
    classifier = load_model('DogCat_Classifier.h5')
    result = classifier.predict(images)
    if result[0][0] >= 0.5:
        prediction = 'dog'
        probability = result[0][0]
        print ("probability = " + str(probability))
        print ("Prediction = " + prediction)
    else:
        prediction = 'cat'
        probability = 1 - result[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + prediction)