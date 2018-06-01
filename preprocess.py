import cv2
import os
import sys
import shutil
import yaml

def histogramEqualization(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    output = cv2.cvtColor(image,cv2.COLOR_YUV2BGR)
    return output

def main():
    with open('./system_config.json') as f:
        config = yaml.load(f)['preprocess']

    indir = config['indir']
    outdir = config['outdir']

    classList =  os.listdir(indir)
    for c in classList:
        cIndir = os.path.join(indir,c)
        cOutdir = os.path.join(outdir,c)

        if os.path.exists(cOutdir):
            shutil.rmtree(cOutdir)
        os.makedirs(cOutdir)

        fnames = [f for f in os.listdir(cIndir)
                  if os.path.isfile(os.path.join(cIndir, f))]
        totalFiles = len(fnames)

        for i in range(0,len(fnames)):
            print('>> Preprocessing class =', c, fnames[i], round(float(i) / (totalFiles - 1), 2) * 100, '%')
            img = cv2.imread(os.path.join(cIndir, fnames[i]))
            img = histogramEqualization(img)
            cv2.imwrite(os.path.join(cOutdir, fnames[i][0:-4]+'.jpg'), img)

if __name__ == '__main__':
    main()
