import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ntpath
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

def getName(filepath):
    head, tail = ntpath.split(filepath)
    return tail

def importDataInfo(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    data = pd.read_csv(os.path.join(path,"driving_log.csv"), names=columns)
    print(getName(data['Center'][0]))
    data["Center"] = data["Center"].apply(getName)
    # print(data.shape)
    # print(data.head())
    # print(data.shape)
    return data

def balanceData(data, display=True):
    nBins = 25
    samplesperbin = 284
    hist, bins = np.histogram(data['Steering'], nBins)
    print(bins)
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        # print(center)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesperbin,samplesperbin))
        plt.show()
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data["Steering"])):
            if data["Steering"][i]>bins[j] and data["Steering"][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesperbin:]
        removeIndexList.extend(binDataList)
    print("Removed Images: ", len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)
    print("Remaining Images", len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesperbin,samplesperbin))
        plt.show()

def loadData(path,data):
    imagespath = []
    steering = []
    for i in range(len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagespath.append(os.path.join(path,'IMG',indexedData[0]))
        # print(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imagespaths = np.asarray(imagespath)
    steering = np.asarray(steering)
    return imagespaths, steering

def augumentImage(imgpath, steering):
    img = mpimg.imread(imgpath)
    #pan
    if np.random.rand() < 0.5:
        pan = iaa.Affine(transalate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    # zoom
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    # Brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)

    # Flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

# imgRe, st = augumentImage('test.jpg',0)
# plt.imshow(img)
# plt.show()

def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# imgRe = augumentImage(mpimg.imread('test.jpg'))
# plt.imshow(img)
# plt.show()

def batchGen(imagespath, steeringList, batchSize, istraining):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagespath) - 1)
            if istraining:
                img, steering = augumentImage(imagespath[index], steeringList[index])
            else:
                img = mpimg.imread(imagespath[index])
                steering = steeringList[index]

            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))

def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))


    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.0001), loss="mse")

    return model