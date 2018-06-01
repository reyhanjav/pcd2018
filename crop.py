import cv2

def getRegion(img):
    padding = 0.15
    rows,cols,_ = img.shape
    dY = round(rows*padding)
    dX = round(cols*padding)
    outp = img[dY:(rows-dY), dX:(cols-dX)]
    return outp
