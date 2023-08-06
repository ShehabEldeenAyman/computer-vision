#author: shehab el deen ayman mounir
import numpy as numpy
from PIL import ImageGrab
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def MakeCoordinates(frame,lines):
    slope,intercept = lines
    print(frame.shape)
    y1 = frame.shape[0]
    y2 = int(y1*3/5)
    x1 = int(y1 - intercept/slope)
    x2 = int(y2 - intercept/slope) 
    return numpy.array([x1,y1,x2,y2])
    
    
    
    
def AverageSlope(frame,lines):
    LeftSide = []
    RightSide = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = numpy.polyfit((x1,x2),(y1,y2),1)
        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            LeftSide.append((slope,intercept))
        else:
            RightSide.append((slope,intercept))
    AverageLeftSide = numpy.average(LeftSide,axis=0)
    AverageRightSide = numpy.average(RightSide,axis=0)  
    LeftLine=MakeCoordinates(frame,AverageLeftSide)
    RightLine=MakeCoordinates(frame,AverageRightSide)
    return numpy.array([LeftLine,RightLine])         



def single_line(frame,lines):
    LeftLineX = []
    LeftLineY = []
    RightLineX = []
    RightLineY = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                LeftLineX.extend([x1, x2])
                LeftLineY.extend([y1, y2])
            else: 
                RightLineX.extend([x1, x2])
                RightLineY.extend([y1, y2])
            
    MinimumY = image.shape[0] * (3 / 5) 
    MaxuimumY = image.shape[0] 
    poly_left = numpy.poly1d(numpy.polyfit(
    LeftLineY,
    LeftLineX,
    deg=1
    ))
    StartLeftX = int(poly_left(MaxuimumY))
    EndLeftX = int(poly_left(MinimumY))
    poly_right = numpy.poly1d(numpy.polyfit(
        RightLineY,
        RightLineX,
        deg=1
    ))
    StartRightX = int(poly_right(MaxuimumY))
    EndRightX = int(poly_right(MinimumY))
    ImageLines = DrawLines(
    image,
    [[
        [StartLeftX, MaxuimumY, EndLeftX, MinimumY],
        [StartRightX, MaxuimumY, EndRightX, MinimumY],
    ]],
    thickness=5,
    )
    return ImageLines
    
    
    

def FrameProcessing(frame):
    gray =cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    blur =cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,100,200)    
    return canny
                

def RegionOfIntrest(frame):
    triangle = numpy.array([[(94,720),(1031,720),(781,349),(362,353)]])
    
    mask = numpy.zeros_like(frame)
    cv2.fillPoly(mask,triangle,(255,255,255))
    masked_frame = cv2.bitwise_and(frame,mask)
    return masked_frame
    

def DrawLines(frame,lines):
    ImageLines = numpy.zeros_like(frame)
    if lines is not None: 
         for line in lines:
             x1,y1,x2,y2 = line.reshape(4)
             cv2.line(ImageLines,(x1,y1),(x2,y2),(255,0,0),5)
    return ImageLines
    

while(True):
    image = numpy.array(ImageGrab.grab(bbox=(0,40,1280,720)))
    lane_screen = numpy.copy(image)
    canny = FrameProcessing(lane_screen)
    crop_frame = RegionOfIntrest(canny)
    lines = cv2.HoughLinesP(crop_frame,2,numpy.pi/180,100,numpy.array([]),minLineLength=40,maxLineGap=40)
    line_image = DrawLines(lane_screen,lines)
    ImageLines = DrawLines(lane_screen,lines)
    final_frame = cv2.addWeighted(lane_screen,0.8,ImageLines,1,1)
    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("final_frame",final_frame)
    cv2.imshow("cropped",crop_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    