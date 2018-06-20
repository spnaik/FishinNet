import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras

train_dir = "../train/"
test_dir = "../test_stg1"

classes = os.listdir(train_dir)
classes = [x for x in os.listdir(train_dir) if not x.startswith('.')]
print (classes)
nTrain_data= []
nVal_data = []
train_labels=[]
val_labels=[]
for c in classes:
    t = [c]*len(os.listdir(train_dir+c+'/'))
    nTrain_data.extend(t)
    train_labels.append(len(os.listdir(train_dir+'/'+c)))
print ('the total number of training data is', len(nTrain_data))
ind = np.arange(8)
print (ind)

#------------uncomment below lines or use Notebook ----------------
#plt.figure()
#plt.bar(ind,train_labels,color=['orange'])
#plt.xticks(ind,classes,rotation=90,fontsize=16)
#plt.yticks(fontsize=16)
#plt.title("Training Classes Distribution",fontsize=17)
#plt.show()
