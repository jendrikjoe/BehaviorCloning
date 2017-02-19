#**Behavioral Cloning** 

##Report




[//]: # (Image References)

---




## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py BehaviourCloning/model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
The neutal network consists of three 5x5 convolution layers followed by two inception layers and three dense ones.

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer after the last dense layer to reduce overfitting. 
The model was trained and validated on different data sets to ensure that the model was not overfitting. 
In addition early stopping was used with a validation data set to avoid overfitting.
The model was tested by evaluating it on a test data set and running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

An Adam optimizer was used for training, therefore the learning rate was not tuned.

####4. Appropriate training data

For training the data provided by Udacity were used, plus two laps driven in the right direction. Due to the selection of images from these sets using the binning technique, they provided a well enough basis for the algorithm to learn.
Training was dones using a training set, a validation set for early stop and a test set for comparison. The splits were the following test/(training + validation + test) = .2, validation/(train + validation) =.1. This provides reasonably well validation and test accuracies, while not cutting to much into the training data. The initial training approach was to use the Adam optimizer using the full training set each epoch.
This led to a model which could drive the first track already.

###Model Architecture and Training Strategy

####1. Solution Design Approach

As image data are processed, convolution layers were used. 

The first approach was simply to stack three of these layers on top of each other, followed by two dense layers. The convolution layers had 5x5 matrices and were followed by MaxPooling with strides of 2x2. The two dense layers were deployed with a width of 1024 and 128 allowing to catch a wide variety of aspects without using to much processing power. Hereby, I started to employ dropout after the last dense layer to avoid overfitting and kept this approach for all further networks. This approach together with the limitation of samples with a similar steering angle allowed already to let the car drive around the track. However, it drove in serpentine lines.
Later, I migrated to the NVIDIA approach after playing but with two inception layers instead of the 3x3 convolution layers.
The final version consists now of three 5x5 convolution layers followed by two inception layers and three dense ones.

This allowed the model to drive reliable and with a minimum of action.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

* A 5x5x24 convolution layer with 2x2 striding (l. 277-279)
* A 5x5x32 convolution layer with 2x2 striding (l. 280-282)
* A 5x5x48 convolution layer with 2x2 striding (l. 283-285)
* An inception layer (l. 287-296)
* A 5x5x32 convolution layer with 2x2 striding (l. 299-302)
* An inception layer (l. 304-313)
* A 1024 dense layer (l. 317-319)
* A 256 dense layer (l. 320-322)
* A 128 dense layer (l. 323-325)

All of the convolution layers use hereby valid padding and both interception layers are connected to the 1024 dense layer.


####3. Creation of the Training Set & Training Process

The basis for the training and validation set were formed using the data provided by Udacity as well as two recorded rounds in the correct direction.
The data were then presorted by splitting them into bins with a width of 0.1 based on their steering angle.
Hereby a maximum of 200 values was added per bin and values with an absolute value of more than 0.5 where added two times. This way, the bias from the data was lowered.
Training was dones using a training set, a validation set for early stop and a test set for comparison. The splits were the following test/(training + validation + test) = .2, validation/(train + validation) =.2. This provides reasonably well validation and test accuracies, while not cutting to much into the training data. The initial training approach was to use the Adam optimizer using the full training set each epoch.


The data were preprocessed the following:
* A grayscale channel was added as a fourth channel to the images
* The grayscale channel was normalised using CLAHE
* All channels were normalised to provide a mean of 0 and a deviation of 1

The training data were augmented in a generator pipeline in the following way:
* With 10% probability each the left or the right image were chosen. If so, the steering angle is change by +.2 for the left image and -.2 for the right one.
* The image is mirrored with a probability of 50%, inverting the image in the width direction and multiplying the steering angle with -1.
* A horizontally shifted version of the data was produced by shifting them by a randomly selected value between -20 and 20 pixels along the horizontal axis. The steering angle was hereby change by the amount -.1*shift/20, to explain the model to drive in the centre of the road.
* A vertically shifted version of the data was produced by shifting them by a randomly selected value between -10 and 10 pixels along the vertical axis. The steering angle was multiplied by (1-shiftVer/100), to take more distant curves into account.
* The image is cropped cutting 10 pixels on the top and bottom and 20 on the left and right to remove any indication of a shift from the image.
* A rotated version of the image is produced by rotating between -10 and 10 deg around (max(y), max(x)/2). The steering angle was hereby change by the amount 
angle/(1.5*25), to take account for the changed situation.


For training an adam optimizer was used to avoid manually training the learning rate. Hereby, an early stop criterion on the validation accuracy with a delta of 0.0005 and a patience of 8 was used to avoid overfitting. To avoid getting stuck at one of the first epochs, 10 epochs are applied without an early stop criterion.