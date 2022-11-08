# Mazen Darwish
# March/April 2021

#====== peripheralUI.py ======#
# HELPER

# IN: Image from the canvas of an arbitrary size. Monocolour with white as the background.
# OUT: MNIST format image of 28x28 size, incl. 4px padding, and black as background.
#
# This file contains all the functions necessary to process the canvas input.
# Canvas input needs to be processed and made to be similar to the training data for maximum accuracy.

import numpy as np
import cv2
import math
import scipy.ndimage as ndimage

#Get the centre of mass in the image (So we can centre the number)
def getShiftCoords(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    xshift = np.round(cols/2.0-cx).astype(int)
    yshift = np.round(rows/2.0-cy).astype(int)

    return xshift, yshift

#Shift the number to the centre
def shift(img,xshift,yshift):
    rows,cols = img.shape
    matrix = np.float32([[1,0,xshift],[0,1,yshift]])
    shifted = cv2.warpAffine(img,matrix,(cols,rows)) #Transforming the image to the centre
    return shifted

#Remove excess whitespace
def cropInput(img):
    #Invert image so that the input digit is white and the background is black (like the MNIST dataset)
    img = 255-img 

    coords = cv2.findNonZero(img) # Find all non-zero points (number)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    return img[y:y+h, x:x+w] # Crop the image to remove all the extra whitespace

#This function converts the canvas image to MNIST format
#MNIST dataset dimensions can be found here: https://paperswithcode.com/dataset/mnist
def convertToMNIST(img):
    rows,cols = img.shape
    #Image needs to fit into a 20x20 pixel box to be similar to the MNIST dataset, however we need to keep aspect ratio
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        img = cv2.resize(img, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        img = cv2.resize(img, (cols, rows))

    #Add padding as entire image needs to be 28X28 
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    img = np.pad(img,(rowsPadding,colsPadding),'constant')

    #Shifting it towards the centre of mass
    xshift, yshift = getShiftCoords(img)
    shifted = shift(img,xshift,yshift)
    return shifted
