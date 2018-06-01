import numpy as np
import cv2
import argparse
import os
import pickle
import yaml

from preprocess import histogramEqualization
from feature import getAverage

def main():
    with open('./system_config.json') as f:
        config = yaml.load(f)['identify']
    classifier = pickle.load( open( config['model'], "rb" ) )
    indir = config['indir']

    imgFiles = os.listdir(indir)

    print "=====Y Predict====="
    for i in imgFiles:
	    # read image
	    imgFilePath = (os.path.join(indir, i))
	    image = cv2.imread(imgFilePath)
	    # preprocess
	    image = histogramEqualization(image)
	    
	    # get feature
	    X = getAverage(image)

	    # predict
	    y_pred = classifier.predict(X, False)

	    # print result
	    print imgFilePath, ' predicted as ', y_pred

if __name__ == '__main__':
	main()
