'''
Created on Feb 19, 2017

@author: jjordening
'''

import cv2
import numpy as np

def minValImage(arr, channel = 0):
    """
    Determines the minimum value of arr in a channel
    """
    return np.min(np.min(np.min(arr[:,:,:,channel], axis=1),axis=1),axis=0)

def maxValImage(arr, channel = 0):
    """
    Determines the maximium value of arr in a channel
    """
    return np.max(np.max(np.max(arr[:,:,:,channel], axis=0),axis=0),axis=0)

def addGrayLayer(image):
    """
    Adds a gray layer as a fourth channel to an image
    
    Input: 
        image
        
    Output:
        an array of images with the channels RGBGray
    """
    return np.concatenate((image, image[:,:,0].reshape((image.shape[0], image.shape[1],1))), axis=2)
    

def applyNormalisation(image):
    """
        Applies a normalisation to an image with the channels RGBGray.
        It applies a CLAHE normalisation to the gray layer and then normalises the
        values such, that they have a mean of 0 and a deviation of 1
        
        Input: 
            image
        
        Output:
            an array of images with the channels RGBGray
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image[:,:,3] = clahe.apply(image[:,:,3])
    image = (2.*(image[:,:] - np.min(np.min(image,axis=0), axis=0))[:,:]/
          (np.max(np.max(image,axis=0), axis=0)-np.min(np.min(image,axis=0), axis=0))) -1
    return image

def preprocessImage(image, transform):
    """
    This function represents the default preprocessing for 
    an image to prepare them for the network
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image = cv2.convertScaleAbs(image, alpha=(1))
    image = addGrayLayer(image)
    image = np.concatenate((image, transform(image)), axis=2)
    return applyNormalisation(image)

def preprocessImages(arr, transform):
    """
    This function represents the default preprocessing for 
    images to prepare them for the network
    """
    return np.array([preprocessImage(image) for image in arr])


def perspectiveTransform(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
        

def shiftImg(arr, horizontal, vertical):
    """
        This function shifts an image horizontally and vertically
        Input:
            horizontal - amplitude of shift in pixels (positive to the left
            negative to the right)
            vertical - aplitude of the ishift in pixels (positive upwards 
            negative downwards)
    """
    arr = arr.copy()
    if(vertical>0):arr = np.concatenate((arr[vertical:,:],np.zeros((vertical,arr.shape[1],arr.shape[2]))), axis=0)
    elif(vertical<0):arr = np.concatenate((np.zeros((np.abs(vertical),arr.shape[1],arr.shape[2])), arr[:vertical,:]), axis=0)
    if(horizontal>0):arr = np.concatenate((arr[:,horizontal:],np.zeros((arr.shape[0],horizontal,arr.shape[2]))), axis=1)
    elif(horizontal<0):arr = np.concatenate((np.zeros((arr.shape[0], np.abs(horizontal),arr.shape[2])), arr[:,:horizontal]), axis=1)
    return arr

def mirrorImage(img):
    """
        This function mirrors the handed image around the y-axis
    """
    return img[:,::-1]

def rotateImage(img, angle):
    """
        Rotates image around the point in the middle of the bottom of the picture by
        angle degrees.
    """
    rotation = cv2.getRotationMatrix2D((img[0].shape[0], img[0].shape[1]/2), angle, 1)
    return cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
    
def rotateImages(arr, angles):
    """
        Rotates multiple images by the given angles.
    """
    arr = [rotateImage(img, angle) for img, angle in zip(arr, angles)]
    return arr

