# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import random
import os
import cv2

# 1 - means crosswalk
# 0 - means other sign type
classIdConversion = {'speedlimit': 0,
              'stop': 0,
              'crosswalk': 1,
              'trafficlight': 0}

testAnnotationsPath = './test/annotations'
testImagesPath = './test/images'
trainAnnotationsPath = './train/annotations'
trainImagesPath = './train/images'

# Object class for storing data of object in the picture
class Object:
    xmax = 0
    xmin = 0
    ymax = 0
    ymin = 0
    name = ''

    def __init__(self, name, xmax, xmin, ymax, ymin):
        self.name = name
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

# Function for cutting object from entire photo based on corner points of rectangle
def CutObjectFromImage(img, startCol, endCol, startRow, endRow):
    return img[startRow:endRow, startCol:endCol]

# Get all of the files from directory of specified type
def GetListOfFiles(pathSelected, fileType):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
                                                files in os.walk(pathSelected) for f in files if f.endswith(fileType)]
# Check Quantity of crosswalks inside of dataset along with other objects
def CheckQuantity(data):
    other = 0
    crosswalk = 0
    for annotation in data:
        for object in annotation['objects']:
            name = object.name
            if classIdConversion[name]:
                crosswalk = crosswalk + 1
            else:
                other = other + 1

    print("Crosswalk", crosswalk, "\nOthers", other)


def GetAnnotationsData(annotationPath):
    annotationsPaths = GetListOfFiles(annotationPath, '.xml')
    annotationList = []
    for a_path in annotationsPaths:
        # Get root of xml file
        root = ET.parse(a_path).getroot()
        annotation = {}
        # Get all of the specified values
        annotation['filename'] = root.find("./filename").text
        annotation['width'] = root.find("./size/width").text
        annotation['height'] = root.find("./size/height").text

        # Finding all posible objects and collect their values
        name = root.findall("./object/name")
        xmin = root.findall("./object/bndbox/xmin")
        ymin = root.findall("./object/bndbox/ymin")
        xmax = root.findall("./object/bndbox/xmax")
        ymax = root.findall("./object/bndbox/ymax")

        # Add all objects to the list
        objectsList = []
        for i in range(len(name)):
            objectsList.append(Object(name[i].text, int(xmax[i].text), int(xmin[i].text), int(ymax[i].text), int(ymin[i].text)))

        annotation['objects'] = objectsList # Add list of the objects to annotation
        annotationList.append(annotation) # Add annotation to list of annotations
    return annotationList

# Function for printing data according to template
def PrintAnnotations(data):
    for annotation in data:
        print(annotation['filename'])
        objectsList = annotation['objects']
        n = len(objectsList)
        print(n)
        for obj in objectsList:
            print(obj.xmin, obj.xmax, obj.ymin, obj.ymax)

# Calculate field of object
def CalculateField(xmin, xmax, ymin, ymax):
    return (xmax-xmin) * (ymax-ymin)

# Loading images based on annotations
def LoadData(path, annotations, train):
    # train object
    data = []
    for annotation in annotations:
        image = cv2.imread(os.path.join(path, annotation['filename']))
        for object in annotation['objects']:
            if train:
                img = CutObjectFromImage(image, object.xmin, object.xmax, object.ymin, object.ymax)
            else:
                img = image
            name = object.name
            fieldPercent = CalculateField(object.xmin, object.xmax, object.ymin, object.ymax) / \
                           (int(annotation['width']) * int(annotation['height']))
            classId = classIdConversion[name]
            # if fieldPercent >= 0.1 and not train:
            #     data.append({'image': img, 'label': classId})
            # if train:
            data.append({'image': img, 'label': classId})

    return data

def BalanceData(data, ratio):
    sampledData = random.sample(data, int(ratio * len(data)))

    return sampledData

def LearnBoVW(data):
    # TODO
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()
    np.save('voc.npy', vocabulary)


def ExtractFeatures(data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        if desc is not None:
            sample.update({'desc': desc})
        else:
            sample.update({'desc': np.zeros((1, 128))})
        # ------------------

    return data

def Train(data):
    desc = []
    labels = []
    for sample in data:
        if sample['desc'] is not None:
            desc.append(sample['desc'].squeeze(0))
            labels.append(sample['label'])

    rf = RandomForestClassifier()
    rf.fit(desc, labels)
    # ------------------

    return rf

def Predict(rf, data):
    # TODO
    for sample in data:
        if sample['desc'] is not None:
            pred = rf.predict(sample['desc'])
            sample['label_pred'] = int(pred)
    # ------------------

    return data



def Evaluate(data):
    y_pred = []
    y_real = []
    n_corr = 0
    n_incorr = 0
    for sample in data:
        if sample['desc'] is not None:
            y_pred.append(sample['label_pred'])
            y_real.append(sample['label'])
            if sample['label_pred'] == sample['label']:
                n_corr += 1
            else:
                n_incorr += 1

    confusion = confusion_matrix(y_real, y_pred)

    # Detection of all types
    # _TPa, _Eba, _Eca, _Eda, _Eab, _TPb, _Ecb, _Edb, _Eac, _Ebc, _TPc, _Edc, _Ead, _Ebd, _Ecd, _TPd = confusion.ravel()
    # print(confusion)
    # accuracy = 100 * (_TPa + _TPb + _TPc + _TPd) / (_TPa + _Eba + _Eca + _Eda + _Eab + _TPb + _Ecb + _Edb + _Eac + _Ebc + _TPc + _Edc + _Ead + _Ebd + _Ecd + _TPd)

    # Binary detection
    if n_incorr:
        _TPa, _Eba, _Eab, _TPb = confusion.ravel()
        accuracy = 100 * (_TPa + _TPb) / (_TPa + _Eba + _Eab + _TPb)
        print("accuracy =", round(accuracy, 2), "%")
    # ------------------
    print("Score = %.2f" % (100*n_corr/max(n_corr+n_incorr, 1)), " %")

    # this function does not return anything
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Getting annotations")
    trainAnnotations = GetAnnotationsData(trainAnnotationsPath)
    testAnnotations = GetAnnotationsData(testAnnotationsPath)

    print("Check data in datasets")
    print("Testing data")
    CheckQuantity(testAnnotations)
    print("Training data")
    CheckQuantity(trainAnnotations)

    print("Annotations data:")
    PrintAnnotations(trainAnnotations)
    trainData = LoadData(trainImagesPath, trainAnnotations, 1)
    trainData = BalanceData(trainData, 1.0)

    if os.path.isfile('voc.npy'):
        print('BoVW is already learned')
    else:
        LearnBoVW(trainData)

    print("Extracting train features")
    trainData = ExtractFeatures(trainData)

    print("Training")
    rf = Train(trainData)

    print("Extracting test data")
    print("Annotations data:")
    PrintAnnotations(testAnnotations)
    testData = LoadData(testImagesPath, testAnnotations, 0)
    testData = BalanceData(testData, 1.0)

    print("Extracting test features")
    trainData = ExtractFeatures(testData)

    print("Testing on testing dataset")
    testData = Predict(rf, testData)
    Evaluate(testData)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
