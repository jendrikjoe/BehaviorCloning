# Load pickled data
import pandas as pd
import tensorflow as tf 
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.utils import shuffle
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from docutils.nodes import image
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

LOADMODEL = False


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
#flags.DEFINE_string('trainingCSV', '../simulator/simulator-linux/driving_log.csv', "training data")
flags.DEFINE_string('trainingCSV', '../simulator/data/data/driving_log.csv', "training data")
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

def preprocessImage(image):
    """
    This function represents the default preprocessing for 
    an image to prepare them for the network
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    image = cv2.convertScaleAbs(image, alpha=(1))
    image = addGrayLayer(image)
    return applyNormalisation(image)

def preprocessImages(arr):
    """
    This function represents the default preprocessing for 
    images to prepare them for the network
    """
    return np.array([preprocessImage(image) for image in arr])
        

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

def generateTrainImagesFromPaths(data, batchSize, inputShape, outputShape):
    """
        The generator function for the training data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    returnArr = np.zeros((batchSize, inputShape[0]-20, inputShape[1]-40, inputShape[2]))
    labels = np.zeros((batchSize, outputShape[0]))
    while 1:
        indices = np.random.randint(0, len(data), batchSize)
        for i,index in zip(range(len(indices)),indices):
            row = data.iloc[index]
            imSelect = np.random.random()
            if(imSelect <.1):
                image = np.array(mpimg.imread(row['right'].strip()))
                label = np.array([min(row['steering']-.2,-1), row['throttle'], row['break']])
            elif(imSelect >.9):
                image = np.array(mpimg.imread(row['left'].strip()))
                label = np.array([min(row['steering']+.2,1), row['throttle'], row['break']])
            else:
                image = np.array(mpimg.imread(row['center'].strip()))
                label = np.array([row['steering'], row['throttle'], row['break']])
            image = preprocessImage(image)
            flip = np.random.random()
            if flip>.5:
                image = mirrorImage(image)
                label[0] *= -1
                
            shiftHor = np.random.randint(-20,21)
            shiftVer = np.random.randint(-10,11)
            image = shiftImg(image, shiftHor, shiftVer)
            label[0] *= (1-shiftVer/100)
            label[0] -= .1*shiftHor/(20)
            image = image[10:-10,20:-20]
            rot = np.random.randint(-10,11)
            image = rotateImage(image, rot)
            # Add a part of the rotated angle, as it is counted counter-clockwise.
            # If you turn counter-clockwise, this looks like the car would be more left
            # and needs to drive to the right -> add some angle 
            # divide it by the maximum of the steering angle in deg ->25
            label[0] += rot/(25*1.5)
            label[0] = min(max(label[0],-1),1)
            returnArr[i] = image
            labels[i] = label
        yield({'input_1': returnArr},{'output': labels[:,0]})
                
def generateTestImagesFromPaths(data, batchSize, inputShape, outputShape):
    """
        The generator function for the validation and test data for the fit_generator
        Input:
        data - an pandas dataframe containing the paths to the images, the steering angle,...
        batchSize, the number of values, which shall be returned per call
    """
    size=0
    returnArr = np.zeros((batchSize, inputShape[0]-20, inputShape[1]-40, inputShape[2]))
    labels = np.zeros((batchSize, outputShape[0]))
    while 1:
        row = data.iloc[size%len(data)]
        image = np.array(mpimg.imread(row['center'].strip()))
        label = np.array([row['steering'], row['throttle'], row['break']])
        image = preprocessImage(image)
        flip = np.random.random()
        if flip>.5:
            image = mirrorImage(image)
            label[0] *= -1
        returnArr[size%batchSize] = image[10:-10,20:-20]
        labels[size%batchSize] = label
        if(size%batchSize==0):
            yield({'input_1': returnArr},{'output': labels[:,0]})
        size+=1
            
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

def customLoss(y_true, y_pred):
    """
        This loss function adds some constraints on the angle to 
        keep it small, if possible
    """
    return K.mean(K.square(y_pred - y_true), axis=-1) #+.01* K.mean(K.square(y_pred), axis = -1)

def main():
    plt.xkcd()
    np.random.seed(0)
    data = pd.read_csv(FLAGS.trainingCSV)
    data['center'] = '../simulator/data/data/'+data['center'].apply(lambda x: x.strip())
    data['right'] = '../simulator/data/data/'+data['right'].apply(lambda x: x.strip())
    data['left'] = '../simulator/data/data/'+data['left'].apply(lambda x: x.strip())
    data2 = pd.read_csv('../simulator/simulator-linux/driving_log.csv', header = None, names=['center','left', 'right', 'steering',
                                                               'throttle', 'break', 'speed'])
    data = data.append(data2)
    dataShuffled = shuffle(data, random_state=0)
    print(len(dataShuffled))
    bins = np.arange(-1.2, 1.3, .1)
    dataNew = pd.DataFrame(columns=['center','left', 'right', 'steering',
                                                               'throttle', 'break', 'speed'])
    print(max(dataShuffled['steering']))
    #normalise the distribution of the values using bins
    # hereby every bin contains maximum 200 values with
    # the rarest bins getting the values added three times:
    indices = dataShuffled.index.get_values()
    for binInd in bins:
        val = 0
        removeArr = []
        for i in indices:
            row = dataShuffled.iloc[i]
            if(binInd-.1 <= row['steering'] < binInd and val<=200):
                val+=1
                if(np.abs(binInd)>.5):
                    #if the value is in a bin with rare values, add it two times
                    dataNew = dataNew.append(row)
                dataNew = dataNew.append(row)
                
            removeArr.append(i)
        np.delete(indices, removeArr)
    del dataShuffled, data
    
    print(len(dataNew))
    
    print(dataNew['steering'].value_counts())
    print(len(dataNew['center']))

    
    dataNew = shuffle(dataNew, random_state = 0)
    plt.figure(1, figsize=(8,4))
    plt.hist(dataNew['steering'], bins=np.arange(-1.2, 1.3, .1))
    #plt.show()
    
    dataTrain, dataTest= train_test_split(dataNew, test_size = .2)
    dataTrain, dataVal= train_test_split(dataTrain, test_size = .2)
    
    imShape = preprocessImage(mpimg.imread(dataTrain['center'].iloc[0])).shape
    
    
    batchSize = 128
    epochBatchSize = 2048
    
    trainGenerator = generateTrainImagesFromPaths(dataTrain, batchSize, imShape, [3])
    valGenerator = generateTestImagesFromPaths(dataVal, batchSize, imShape, [3])
    stopCallback = EarlyStopping(monitor='val_loss', patience = 8, min_delta = 0.0005)
    checkCallback = ModelCheckpoint('model.ckpt', monitor='val_loss', save_best_only=True)
    visCallback = TensorBoard(log_dir = './logs')
    if LOADMODEL:
        endModel = load_model('initModel.h5', custom_objects={'customLoss':customLoss})
        endModel.fit_generator(trainGenerator, callbacks=[stopCallback, checkCallback, visCallback], nb_epoch=20, samples_per_epoch=epochBatchSize, 
                               max_q_size=128,validation_data = valGenerator, nb_val_samples=len(dataVal))
        endModel.load_weights('model.ckpt')
        endModel.save('model.h5')
        
        
    else:
        inpC = Input(shape=(imShape[0]-20, imShape[1]-40, imShape[2]))
        xC = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2))(inpC)
        xC = BatchNormalization()(xC)
        xC = Activation('relu')(xC)
        xC = Convolution2D(36, 5,5 , border_mode='valid', subsample=(2,2))(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('relu')(xC)
        xC = Convolution2D(48, 5,5 , border_mode='valid',subsample=(2,2))(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('relu')(xC)
        
        xC1 = Convolution2D(32, 1,1 , border_mode='same')(xC)
        xC1 = Convolution2D(8, 5,5 , border_mode='same')(xC1)
        xC2 = Convolution2D(32, 1,1 , border_mode='same')(xC)
        xC2 = Convolution2D(8, 3,3 , border_mode='same')(xC2)
        xC3 = Convolution2D(8, 1,1 , border_mode='same')(xC)
        xC4 = AveragePooling2D(pool_size =(3,3), strides=(1,1), border_mode='same')(xC)
        xC4 = Convolution2D(8, 1,1 , border_mode='same')(xC4)
        xC = merge([xC1, xC2, xC3, xC4], mode='concat', concat_axis=3)
        xC = BatchNormalization()(xC)
        xC = Activation('relu')(xC)
        xOut1 = Flatten()(xC)
        
        xC = Convolution2D(32, 5,5 , border_mode='valid')(xC)
        xC = MaxPooling2D(pool_size=(2,2))(xC)
        xC = BatchNormalization()(xC)
        xC = Activation('relu')(xC)
        
        xC6 = Convolution2D(32, 1,1 , border_mode='same')(xC)
        xC6 = Convolution2D(4, 5,5 , border_mode='same')(xC6)
        xC7 = Convolution2D(32, 1,1 , border_mode='same')(xC)
        xC7 = Convolution2D(4, 3,3 , border_mode='same')(xC7)
        xC8 = Convolution2D(4, 1,1 , border_mode='same')(xC)
        xC9 = AveragePooling2D(pool_size =(3,3), strides=(1,1), border_mode='same')(xC)
        xC9 = Convolution2D(4, 1,1 , border_mode='same')(xC9)
        xInc2 = merge([xC6, xC7, xC8, xC9], mode='concat', concat_axis=3)
        xInc2 = BatchNormalization()(xInc2)
        xInc2 = Activation('relu')(xInc2)
        xOut2 = Flatten()(xInc2)
        
        xOut = Lambda(lambda x : K.concatenate(x, axis=1))([xOut1, xOut2])
        xOut = Dense(1024)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('relu')(xOut)
        xOut = Dense(256)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('relu')(xOut)
        xOut = Dense(128)(xOut)
        xOut = BatchNormalization()(xOut)
        xOut = Activation('relu')(xOut)
        xOut = Dropout(.3)(xOut)
        xOut = Dense(1, name = 'output')(xOut)
        
        endModel = Model(inpC, xOut)
        endModel.compile(optimizer='adam', loss=customLoss, metrics=['mse'])
        # run a separate generator to make sure not to get stuck at the first step.
        endModel.fit_generator(trainGenerator, callbacks = [visCallback], nb_epoch=10, samples_per_epoch=epochBatchSize, 
                               max_q_size=128, validation_data = valGenerator, nb_val_samples=len(dataVal))
        endModel.fit_generator(trainGenerator, callbacks = [stopCallback, checkCallback,visCallback], nb_epoch=40, samples_per_epoch=epochBatchSize, 
                               max_q_size=128, validation_data = valGenerator, nb_val_samples=len(dataVal))
        endModel.load_weights('model.ckpt')
        endModel.save('initModel.h5')
        endModel.save('model.h5')
        
    endModel = load_model('model.h5', custom_objects={'customLoss':customLoss})
    print(endModel.evaluate_generator(valGenerator, val_samples=len(dataVal)))
    print(endModel.evaluate_generator(generateTestImagesFromPaths(dataTest, batchSize, imShape, [3]), val_samples=len(dataTest)))

if __name__ == '__main__':
    main()
















