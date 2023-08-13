import cv2
import numpy as np
from object_detection import ObjectDetection
import math
from PIL import *
from PIL import ImageGrab
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import time
import argparse
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from numpy import asarray
import ipyplot
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer


file_object = open("data.txt","a+")

class frame_data:
    def __init__(self,id,name,frame_number,type,file_name):
        self.id = id 
        self.name = name
        self.frame_number = frame_number
        self.type = type
        self.file_name = file_name

#cap = cv2.VideoCapture(0)
file_name="test.m4v"
cap = cv2.VideoCapture(file_name)
frame_counter = 0
frame_list = []
type = ''

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[6]
#tracker = cv2.cv2.Tracker_create(tracker_type)
#tracker = cv2.TrackerMOSSE_create()

is_tracking=False
success=False
object_name=''
is_identified = False


od = ObjectDetection()


classes = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light"
,"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse"
,"sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag"
"tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat"
,"baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass"
,"cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli"
,"carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed"
,"diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone"
,"microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors"
,"teddy bear","hair drier","toothbrush"]


def drawBox(img,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img,'tracking',(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    #cv2.putText(img,'testing',(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cropped_image=img[y:y+h,x:x+w]
    cv2.imshow("cropped",cropped_image)
    listOfGlobals = globals()
    object_name='123'
    
    if(listOfGlobals['is_identified']==False):
        (class_ids, scores, boxes) = od.detect(cropped_image)
        object_name = class_ids[0]
        print(class_ids)
        print(object_name)
        print(classes[class_ids[0]])
        listOfGlobals['is_identified']=True
        listOfGlobals['object_name'] = classes[class_ids[0]]
        #classes[object_name[0]]
        #cv2.putText(img,'testing',(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(img,str(object_name),(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)
     

while True:

    frame_counter = frame_counter+1
    #print('frame: ',frame_counter)
    
    timer = cv2.getTickCount()
    success,img = cap.read()
    if(is_tracking):
        success,img = cap.read()
        success,bbox = tracker.update(img)
        if(success):
            drawBox(img,bbox)
        else:
            cv2.putText(img,'tracking lost',(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            #pass


    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img,str(int(fps)),(75,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("Tracking",img)

    if(cv2.waitKey(1) & 0xff ==ord('t')):
        is_tracking=True
        tracker = cv2.TrackerCSRT_create()
        success,img = cap.read()
        bbox = cv2.selectROI("tracking",img,False)
        tracker.init(img,bbox)
        print('tracking')
        
    if(is_tracking==True):
        frame_list.append(frame_data(0,'test subject',frame_counter,object_name)) 

    if(cv2.waitKey(25) & 0xff ==ord('q')):
        cv2.destroyAllWindows()
        for obj in frame_list:
            record = 'frame: ',obj.id,' ',obj.name,' ',obj.frame_number,' ',obj.type
            print(object_name)
            file_object.write(str(obj.id))
            file_object.write('\t')
            file_object.write(str(obj.name))
            file_object.write('\t')
            file_object.write(str(obj.frame_number))
            file_object.write('\t')
            file_object.write(str(obj.type))
            file_object.write('\n')

        file_object.close()

        break