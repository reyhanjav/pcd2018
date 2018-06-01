import cv2
import sys
import os
import csv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
filename = args["image"]

def deteksiTomat():
        cascade_file = "xml/cascade.xml"

        cascade = cv2.CascadeClassifier(cascade_file)
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cv2.imshow("gray", gray)

        faces = cascade.detectMultiScale(gray,scaleFactor = 2.9,minNeighbors = 9,minSize = (30, 30))

        idx = 0
        os.system('rmdir /s /q pic')
        os.mkdir('pic')
        os.chdir("pic")
        cv2.imwrite('input.jpg',image)
        for (x, y, w, h) in faces:
                roi=image[y:y+h,x:x+w]   
                cv2.imwrite(str(idx) + '.jpg', roi)
                with open('input.csv', 'a') as f:
                        f.write('{0},{1},{2},{3},{4}\n'.format(str(idx) + '.jpg', x, y, w, h))
                idx = idx + 1

        for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        cv2.imshow("res", image)
        cv2.waitKey(0)

deteksiTomat()

