#!/usr/bin/python

#face_detect.py

#Face Detection using OpenCV. Based on sample code from:
#http://python.pastebin.com/m76db1d6b

#Usage: python face_detect.py <image_file>

import sys, os
from opencv.cv import *
from opencv.highgui import *
import Image, ImageDraw

CLASSIFIER = '/usr/share/doc/opencv-doc/examples/haarcascades/haarcascades/haarcascade_frontalface_default.xml'
def detectObjects(image):
    """Converts an image to grayscale and prints the locations of any face found"""
    grayscale = cvCreateImage(cvSize(image.width, image.height),8,1)
    cvCvtColor(image, grayscale, CV_BGR2GRAY)
    storage = cvCreateMemStorage(0)
    cvEqualizeHist(grayscale, grayscale)
    cascade = cvLoadHaarClassifierCascade(
        CLASSIFIER,
        cvSize(1,1))
    faces = cvHaarDetectObjects(grayscale, cascade, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING,       cvSize(50,50))
    if faces.total > 0:
        for f in faces:
            x1,y1,x2,y2=f.x,f.y,f.x+f.width,f.y+f.height
            print("[(%d,%d)->(%d,%d)]" % (x1,y1,x2,y2))
            print_rectangle(x1,y1,x2,y2) # call to python pil

def print_rectangle(x1,y1,x2,y2): #function to modify the imae
    im = Image.open(sys.argv[1])
    draw = ImageDraw.Draw(im)
    draw.rectangle([x1,y1,x2,y2])
    im.save(sys.argv[1])

def main():
    image = cvLoadImage(sys.argv[1]);
    detectObjects(image)

if __name__ == "__main__":
    main();
