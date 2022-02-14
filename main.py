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



# 0 - means other sign type
classIdConversion = {'speedlimit': 0,
              'stop': 0,
              'crosswalk': 1,
              'trafficlight': 0}

testAnnotationsPath = './test/annotations'
testImagesPath = './test/images'
trainAnnotationsPath = './train/annotations'
trainImagesPath = './train/images'

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


def GetListOfFiles(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
                                                files in os.walk(root) for f in files if f.endswith(file_type)]

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
        root = ET.parse(a_path).getroot()
        annotation = {}
        annotation['filename'] = root.find("./filename").text
        annotation['width'] = root.find("./size/width").text
        annotation['height'] = root.find("./size/height").text

        # Finding all posible objects
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

def LoadData(path, annotations):
    # train object
    data = []
    for annotation in annotations:
        image = cv2.imread(os.path.join(path, annotation['filename']))
        for object in annotation['objects']:
            name = object.name
            classId = classIdConversion[name]
            data.append({'image': image, 'label': classId})

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
        # compute descriptor and add it as "desc" entry in sample
        # TODO
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        if desc is not None:
            sample.update({'desc': desc})
        else:
            sample.update({'desc': np.zeros((1, 128))})
        # ------------------

    return data

def Train(data):
    # TODO
    # Uff
    # clf = RandomForestClassifier(128)
    # x_matrix = np.empty((1,128))
    # y_vector = []
    # for sample in data:
    #     y_vector.append(sample['label'])
    #     x_matrix = np.vstack((x_matrix, sample['desc']))
    # clf.fit(x_matrix[1:], y_vector)
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
            # sample.update({'label_pred': rf.predict(sample['desc'])[0]})
            pred = rf.predict(sample['desc'])
            sample['label_pred'] = int(pred)
    # ------------------

    return data


def Evaluate(data):
    # TODO
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

    _TPa, _Eba, _Eca, _Eda, _Eab, _TPb, _Ecb, _Edb, _Eac, _Ebc, _TPc, _Edc, _Ead, _Ebd, _Ecd, _TPd = confusion.ravel()
    print(confusion)
    accuracy = 100 * (_TPa + _TPb + _TPc + _TPd) / (_TPa + _Eba + _Eca + _Eda + _Eab + _TPb + _Ecb + _Edb + _Eac + _Ebc + _TPc + _Edc + _Ead + _Ebd + _Ecd + _TPd)
    print("accuracy =", round(accuracy, 2), "%")
    # ------------------
    print("Score = %.3f" % (n_corr/max(n_corr+n_incorr, 1)))

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
    trainData = LoadData(trainImagesPath, trainAnnotations)
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
    testData = LoadData(testImagesPath, testAnnotations)
    testData = BalanceData(testData, 1.0)

    print("Extracting test features")
    trainData = ExtractFeatures(testData)

    print("Testing on testing dataset")
    testData = Predict(rf, testData)
    Evaluate(testData)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
