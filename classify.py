import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import glob
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from subprocess import check_output


#print(check_output(["ls", "./input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#read data
data = pd.read_csv('./feature/feature.csv')

#split 70% training 30% testing
train, test = train_test_split(data, test_size=0.3)
#print(test)

print("starting SVM..")

clf = svm.SVC(kernel='rbf',gamma=0.1, C=100 )

features = train[[
		  "r", "g", "b"
                  ]]
target = train['class'].astype(int)
clf = clf.fit(features, target)

#save model to disk
filename = 'svm_model.p'
pickle.dump(clf, open(filename, 'wb'),protocol=2)

#load model svm
#clf = pickle.load(open(filename, 'rb'))


#akurasi
features2 = test[["r", "g", "b"
                  ]]
target2 = test['class'].astype(int)

print("Akurasi :",clf.score(features2, target2))

pred = clf.predict(features2)

df = pd.DataFrame({'Actual': target2, 'Predicted': pred})
print (df)
#print(data)
#if (pred == 1):
#    print("belum matang")
print("selesai")
