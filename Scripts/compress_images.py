
from glob import glob                                                           
import cv2
from PIL import Image

 
jpgs = glob('*.jpg')

for j in jpgs:
    img = Image.open(j)
    img = img.resize((300,160),Image.ANTIALIAS) 
    img.save("/Users/sonali/Documents/insight/compress_image/"+j,optimize=True,quality=95)
