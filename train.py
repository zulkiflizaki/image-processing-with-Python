"""Image Training"""
# pylint:disable=no-member

import os
import cv2 as cv
import numpy as np

objects = ['Building', 'Food', 'Other', 'People']
DIR = 'Images/train/'

# using haar_cascade library
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    """Function printing python version."""
    for object in objects:
        path = os.path.join(DIR, object)
        label = object.index(object)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            object_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in object_rect:
                object_roi = gray[y:y+h, x:x+w]
                features.append(object_roi)
                labels.append(label)

create_train()
print('Training done ---------------')
print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('object_trained.yml')
np.save('object_features.npy', features)
np.save('object_labels.npy', labels)
