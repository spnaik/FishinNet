
from glob import glob                                                           
import cv2 
#jpgs = glob('/Users/sonali/Documents/insight/rohandata/**/*.jpg',recursive=True)
jpgs = glob('/home/paperspace/Documents/insight/images/*.jpg')
for j in jpgs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'png', img)
