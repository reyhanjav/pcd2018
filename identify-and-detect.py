import numpy as np
import cv2

from operator import itemgetter
from histogram import histEqualizeAll
from kelas import saveClass, loadClass, loadDataTrain, loadInput, saveOutput
from classifier import knn
from color_average import getAverage
from output import showOutput


def inputImage(input_directory='detected/',process=1,x=loadInput()):
    outp = []
    for i in x:
        k = 7
        img = cv2.imread(input_directory+i[0])
        img = histEqualizeAll(img)
        kelas = knn(getAverage(img),k,process)
        i.append(kelas)
        outp.append(i)
    saveOutput(outp)
    showOutput('detected/input.jpg')
    #cv2.imshow('citra tomat', img)
    #print('tingkat kematangan citra tomat yang anda masukkan: ' + output)
    #cv2.waitKey(0)

inputImage()

