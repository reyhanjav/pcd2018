import sys
import os
import cv2
import csv
import numpy as np
import yaml

def getAverage(image):
    average_color_per_row = np.average(image, axis = 0)
    average_color = np.round(np.average(average_color_per_row, axis = 0))
    output = [average_color[0], average_color[1],
    		  average_color[2]]
    return output

def main():

    with open('./system_config.json') as f:
        config = yaml.load(f)['feature']

    indir = config['indir']
    outdir = config['outdir']
    features =  [['r','g','b','class']]
    classList =  os.listdir(indir)

    for c in classList:
        if c == '.gitignore':
            continue
        cIndir = os.path.join(indir,c)

        fnames = [f for f in os.listdir(cIndir)
                  if os.path.isfile(os.path.join(cIndir, f))]
        totalFiles = len(fnames)

        for i in range(0,len(fnames)):
            print '>> Extract feature class =', c, fnames[i], round(float(i)/(totalFiles-1),2)*100, '%'
            img = cv2.imread(os.path.join(cIndir, fnames[i]))
            features.append(getAverage(img)+[c])

        # dump feature
        with open(outdir,'wb')as csvfile:
        	writer = csv.writer(csvfile)
        	writer.writerows(features)

if __name__ == '__main__':
	main()
