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
def GetAnnotationsData(ann_path, img_path):
    annotationsPaths = GetListOfFiles(ann_path, '.xml')
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testAnnotationsPath = './test/annotations'
    testImagesPath = './test/images'
    trainAnnotationsPath = './train/annotations'
    trainImagesPath = './train/images'

    # train object
    trainSet = GetAnnotationsData(trainAnnotationsPath, trainImagesPath)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
