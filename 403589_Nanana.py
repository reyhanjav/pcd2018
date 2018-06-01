from __future__ import print_function

import numpy as np
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

fruit_images = []
labels = []
for fruit_dir_path in glob.glob("D:\TUGAS\Arsip Tugas Kuliah\SEMESTER 6\PPCD\detektif-tomat-2-master\output\preprocessed\*"):
    fruit_label = fruit_dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, 0)

        image = cv2.resize(image, (40, 40))

        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
print(id_to_label_dict)

label_ids = np.array([label_to_id_dict[x] for x in labels])

SZ=20
bin_n = 16 # Number of bins

winSize = (40,40)
blockSize = (20,20)
blockStride = (10,10)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

images_hogged = ([hog.compute(i) for i in fruit_images])
#im = hog.compute(fruit_images)

X_train, X_test, y_train, y_test = train_test_split(images_hogged, label_ids, test_size=0.25, random_state=42)
X_tr = np.array(X_train, np.float32)
y_tr = np.array(y_train, np.int32)
X_te = np.array(X_test, np.float32)
y_te = np.array(y_test, np.int32)


# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# Set SVM Kernel to Radial Basis Function (RBF)
#svm.setKernel(cv2.ml.SVM_RBF)
# Set parameter C
#svm.setC(C)
# Set parameter Gamma
#svm.setGamma(gamma)

# Train SVM on training data
svm.trainAuto(X_tr, cv2.ml.ROW_SAMPLE, y_tr)

# Save trained model
svm.save("apples_svm_model.yml")
'''
svm = cv2.ml.SVM_load('apples_svm_model.yml')

pred = svm.predict(X_te)[1]
'''

a = []
for i in range(len(pred)):
    a.append(int(pred[i][0]))
mask = a == y_te
correct = np.count_nonzero(mask)
print("svm model accuracy with HOG descriptor:", end=" ")
print(correct*100.0/pred.size)

validation_fruit_images = []
validation_labels = []

for fruit_dir_path in glob.glob("D:\TUGAS\Arsip Tugas Kuliah\SEMESTER 6\PPCD\detektif-tomat-2-master\output\validation\*"):
    fruit_label = fruit_dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, 0)

        image = cv2.resize(image, (40, 40))

        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)

validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])

validation_hogged = ([hog.compute(i) for i in validation_fruit_images])

V_test = np.array(validation_hogged, np.float32)
V_label = np.array(validation_label_ids, np.int32)

pred1 = svm.predict(V_test)[1]

b = []
for i in range(len(pred1)):
    b.append(int(pred1[i][0]))
mask1 = b == V_label
correct1 = np.count_nonzero(mask1)
print("svm validation accuracy with HOG descriptor:", end=" ")
print(correct1*100.0/pred1.size)
