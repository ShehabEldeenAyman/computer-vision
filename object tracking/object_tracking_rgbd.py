import cv2
import numpy as np
from object_detection import ObjectDetection
import math
from kalmanfilter import KalmanFilter
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


# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")
#cap = cv2.VideoCapture("traffic.jpg")

# Initialize count
count = 0
center_points_prev_frame = []
center_points_prev_frame_depth = []

tracking_objects = {}
tracking_objects_depth_points ={}
track_id = 0

kf = KalmanFilter()

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


while True:
    
    depth_capture = np.array(ImageGrab.grab(bbox=(4,29,400,329)))
    colorFrame_capture = np.array(ImageGrab.grab(bbox=(401,35,1202,636)))
    colorFrame_capture = cv2.cvtColor(colorFrame_capture, cv2.COLOR_BGR2RGB) #/////////important line should not be commented
    depth_capture = cv2.cvtColor(depth_capture, cv2.COLOR_BGR2GRAY)
    
    frame = colorFrame_capture
    
    #ret, frame = cap.read()
    count += 1
    #if not ret:
        #break

    # Point current frame
    center_points_cur_frame = []
    center_points_cur_frame_depth_point = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box,i in zip(boxes,class_ids):
        (x, y, w, h) = box
        cx = int(((x + x + w) / 2))
        cy = int(((y + y + h) / 2))
        
        
        dx = int((((x + x + w) / 2)/2))
        dy = int((((y + y + h) / 2)/2))
        
        
        center_points_cur_frame.append((cx, cy))
        center_points_cur_frame_depth_point.append((dx,dy))
        #print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame,classes[i],(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(depth_capture, (int(x/2), int(y/2)), (int((x + w)/2), int((y + h)/2)), (0, 255, 0), 2)
        cv2.circle(depth_capture, (dx,dy), 5, (0, 0, 255), 4)
        print("depth: ",dx," , ",dy )
        print(classes[i])

        
        

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        counter_p=0
        counter_p2=0
        for pt,pt_d in  zip(center_points_cur_frame, center_points_cur_frame_depth_point):
            for pt2,pt2_d in zip(center_points_prev_frame,center_points_prev_frame_depth): ####################continue from here
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                distance_d = math.hypot(pt2_d[0] - pt_d[0], pt2_d[1] - pt_d[1])
                #distance_depth = ()
                print (distance)
                print("pt: ",pt)
                counter_p2=counter_p2+1

                if distance < 20:
                    print("distance: ",distance)
                    tracking_objects[track_id] = pt
                    tracking_objects_depth_points[track_id]= pt
                    track_id += 1
                
                counter_p2=0
                counter_p = counter_p+1
                    
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        
        tracking_objects_depth_copy = tracking_objects_depth_points.copy()
        center_points_cur_frame_depth_copy  = center_points_cur_frame_depth_point.copy()
        

        for (object_id, pt2),(object_id_d,pt2_d) in zip(tracking_objects_copy.items(),tracking_objects_depth_copy.items()):
            object_exists = False
            for pt,pt_d in zip(center_points_cur_frame_copy,center_points_cur_frame_depth_copy):
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                distance_d = math.hypot(pt2_d[0]-pt_d[0],pt2_d[1]-pt_d[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    tracking_objects_depth_points[object_id_d] = pt
                    object_exists = True
                    
                    predicted = kf.predict(pt2[0], pt2[1])
                    predicted = kf.predict(pt[0], pt[1])
                    cv2.circle(frame, (predicted[0], predicted[1]), 5, (255, 0, 0), 4)
                    cv2.putText(frame, str(object_id), (predicted[0], predicted[1] - 7), 0, 1, (0, 0, 255), 2)
                    
                    
                    
                    
                    
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue
                    
                    if pt_d in center_points_cur_frame_depth_point:
                        center_points_cur_frame_depth_point.remove(pt)
                    continue
                        

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)
                tracking_objects_depth_points.pop(object_id)

        # Add new IDs found
        for pt,pt_d in zip(center_points_cur_frame,center_points_cur_frame_depth_point):
            tracking_objects[track_id] = pt
            tracking_objects_depth_points[track_id] = pt_d
            track_id += 1

    for object_id, pt in tracking_objects.items():
        
        
        

        
        
        cv2.circle(frame, pt, 5, (0, 0, 255), 4)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)


    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)


    cv2.imshow("Frame", frame)
    cv2.imshow("depth frame",depth_capture)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()
    center_points_prev_frame_depth = center_points_cur_frame_depth_point.copy()

    key=cv2.waitKey(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()