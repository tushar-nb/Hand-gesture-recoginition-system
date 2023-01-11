# Hand gesture recognition using neural networks

## By

Tanmai N [PES1UG20CS601]
Tushar N Borkade [PES1UG20CS608]
Ravi Kiran [PES1UG20CS580]

## Problem Statement

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV/Laptop. Each gesture can be set to a specific command:

- Thumbs up
- Thumbs down
- Left swipe
- Right swipe  
- Stop


## Understanding the Dataset

The training data consists of a few hundred images categorised into one of the five classes. These images have been recorded by various people performing one of the five gestures in front of a webcam.

The data is in a folder file. The folder contains a 'train' and a 'val' folder with two CSV files for the two folders.

These folders are in turn divided into subfolders where each subfolder represents a image of a particular gesture.
 (link for the dataset: https://drive.google.com/drive/folders/1LfO_ccc6ACeXAFNvZ9a47n_BWSy2Hs-X?usp=share_link)

## Two Architectures: 2D Convs and RNN Stack

After understanding and acquiring the dataset, the next step is to try out different architectures to solve this problem. 

For analysing images using neural networks, two types of architectures are used commonly. 

One is the standard **CNN + RNN** architecture in which you pass the images through a CNN which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN. 

*Note:*
 - GRU (Gated Recurrent Unit) or LSTM (Long Short Term Memory) can be used for the RNN


## Data Preprocessing

We can apply several of the image procesing techniques for each of image.

### Resize

 We will convert each image of the train and test set into a matrix of size 96*96

### Cropping

Given that one of the data set is of rectangualr shape, we will crop that image to 96*96, this is different to resize, while resize changes the aspect ratio of rectangular image. In cropping we will center crop the image to retain the middle of the frame.

### Normalization

We will use mean normaliztion for each of the channel in the image.

## Data Agumentation

We have a total of 600+ for test set and 100 sampels for validation set. We will increase this 2 fold by usign a simple agumentiaton technique of affine transforamtion.

### Affine Transformation

In affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the transformation matrix, we need three points from input image and their corresponding locations in output image. Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.


We will perform a same random affine transform for all the images in the frameset. This way we are generating new dataset from existing dataset.


## Generators

**Understanding Generators**: As you already know, in most deep learning projects you need to feed data to the model in batches. This is done using the concept of generators. 

Creating data generators is probably the most important part of building a training pipeline. Although libraries such as Keras provide builtin generator functionalities, they are often restricted in scope and you have to write your own generators from scratch. In this project we will implement our own cutom generator, our generator will feed batches of images. 

Let's take an example, assume we have 23 samples and we pick batch size as 10.

In this case there will be 2 complete batches of ten each
- Batch 1: 10
- Batch 2: 10
- Batch 3: 3

The final run will be for the remaining batch that was not part of the the full batch. 

Full batches are covered as part of the for loop the remainder are covered post the for loop.

Note: this also covers the case, where in batch size is day 30 and we have only 23 samples. In this case there will be only one single batch with 23 samples.



# Implementation 

## 2D Convolutional Network, or Conv2D

Now, lets implement a 2D convolutional Neural network on this dataset. To use 2D convolutions, we first extract every image's: width, height,channels. Channels represents the slices of Red, Green, and Blue layers.


Lets create the model architecture. The architecture is described below:

While we tried with multiple ***filter size***, bigger filter size is resource intensive and we have done most experiment with 3*3 filter

We have used **Adam** optimizer with its default settings.
We have additionally used the ReduceLROnPlateau to reduce our learning alpha after 2 epoch on the result plateauing.


## Model #1

Build a 2D convolutional network, based loosely on C2D.

## Model #2

Build a 2D convolutional network, aka C2D.

## Model #3

Custom model

## Model #4
```python
        model = Sequential()

        #Flatten Layers
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        #softmax layer
        model.add(Dense(5, activation='softmax'))
```

Model Summary


## Model #5

Input and Output layers:

- One Input layer with dimentions 30, 120, 120, 3
- Output layer with dimentions 5

Convolutions :

- Apply 4 Convolutional layer with increasing order of filter size (standard size : 8, 16, 32, 64) and fixed kernel size = (3, 3, 3)
- Apply 2 Max Pooling layers, one after 2nd convolutional layer and one after fourth convolutional layer.

MLP (Multi Layer Perceptron) architecture:

- Batch normalization on convolutiona architecture
- Dense layers with 2 layers followed by dropout to avoid overfitting

```python
        model = Sequential()

        #Flatten Layers
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        #softmax layer
        model.add(Dense(5, activation='softmax'))
```

Model Summary


Model 5 gave us **test accuracy of 78% and validation accuracy of 80%** using all the 30 frames. The same model is submitted for the review. 
While we did try model lesser frames by using even frames but we felt more comfortable using full frame. Cropping and other preprocessing also did not affect much on the final accuracy.
