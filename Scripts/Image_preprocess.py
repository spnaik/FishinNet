from __future__ import division, print_function
import ujson as json
import os
import utils
from utils import *
from shutil import copyfile
from keras.utils import get_file
from sklearn.metrics import log_loss
import pandas as pd
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
def get_image_vector(image,size=(64,64)):
    return cv2.resize(image,size).flatten()
def extract_color_histogram(image,bins = (8,8,8)):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,256,0,256,0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()
train_dir = "../train/"
test_dir = "../test_stg1"
classes = os.listdir(train_dir)
classes = [x for x in os.listdir(train_dir) if not x.startswith('.')]
image_path_list = []
for c in classes:
    fish_images = [train_dir+c+'/'+item for item in os.listdir(train_dir+c+'/')]
    image_path_list.extend(fish_images)

labels = []
for c in classes:
    l = [c]*len(os.listdir(train_dir+c+'/'))
    labels.extend(l)
labels = LabelEncoder().fit_transform(labels)
features = []
for i,image_path in enumerate(image_path_list):
    image = cv2.imread(image_path)
    hist = extract_color_histogram(image)
    features.append(hist)
    if(i%1000==0):
        print(str(i)+ "  completed")
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size = 0.30, random_state = 42)
model = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
model.fit(X_train,y_train)
accuracy = model.score(X_test,y_test)
print ('accuracy is',(accuracy))
preds_validation = model.predict_proba(X_test)
log_loss(y_test,preds_validation)
