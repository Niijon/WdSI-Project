# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import random
import os
import cv2



classIdConversion = {'speedlimit': 0,
              'stop': 1,
              'crosswalk': 2,
              'trafficlight': 3}

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainAnnotations = GetAnnotationsData(trainAnnotationsPath)
    print("Annotations data:")
    PrintAnnotations(trainAnnotations)
    trainData = LoadData(trainImagesPath, trainAnnotations)
    trainData = BalanceData(trainData, 1.0)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
