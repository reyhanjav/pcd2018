import cv2
import sys
import os
import csv
from kelas import loadOutput

def showOutput(path):
    data = loadOutput()
    image = cv2.imread(path)
    for i in data:
        if i[5] == 'matang':
            cv2.rectangle(image, (int(i[1]), int(i[2])), (int(i[1]) + int(i[3]), int(i[2]) + int(i[4])), (0, 0, 255), 2)
        elif i[5] =='setengah-matang':
            cv2.rectangle(image, (int(i[1]), int(i[2])), (int(i[1]) + int(i[3]), int(i[2]) + int(i[4])), (0, 125, 255), 2)
        elif i[5] =='mentah':
            cv2.rectangle(image, (int(i[1]), int(i[2])), (int(i[1]) + int(i[3]), int(i[2]) + int(i[4])), (0, 255, 0), 2)
    cv2.imshow("res", image)
    cv2.waitKey(0)

