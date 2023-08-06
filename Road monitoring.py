#author: shehab el deen ayman mounir
import tensorflow as tensorflow
import numpy as numpy
import argparse
import cv2
import os
import math
from PIL import ImageGrab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yolo.networkConfigiration as config
from yolo.yolo_v2 import yolo_v2

point_list_x = []
point_list_y = []  
was_connected = []

class Detector(object):
    
    
    
    
    
    def __init__(self, yolo, weights_file):
        self.yolo = yolo
        self.classes = config.CLASSES
        self.num_classes = len(self.classes)
        
        
        self.BatchSize = config.BATCH_SIZE
        self.Anchor = config.ANCHOR
        self.BoxesPerCell = config.BOX_PRE_CELL
        self.ImageSize = config.IMAGE_SIZE
        self.Threshhold = config.THRESHOLD
        self.CellSize = config.CELL_SIZE
        
        
        self.sess = tensorflow.Session()
        self.sess.run(tensorflow.global_variables_initializer())
        print('Restore weights from: ' + weights_file)
        self.saver = tensorflow.train.Saver()
        self.saver.restore(self.sess, weights_file)
        
        
    def ScreenDetect(self):
    	while True:

    		screen = numpy.array(ImageGrab.grab(bbox=(0,40,800,600)))
    		screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
    		lane_screen = numpy.copy(screen)
    		canny = self.FrameProcessing(lane_screen)
    		crop_frame = self.RegionOfIntrest(canny)
    		lines = cv2.HoughLinesP(crop_frame,2,numpy.pi/180,100,numpy.array([]),minLineLength=40,maxLineGap=40)
    		line_image = self.DrawLines(lane_screen,lines)
    		ImageLines = self.DrawLines(lane_screen,lines)
    		final_frame = cv2.addWeighted(lane_screen,0.8,ImageLines,1,1)
    		final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)

    		result = self.ObjectDetection(screen)
            
    		self.draw(screen,result)
    		cv2.imshow('Image', screen)
    		cv2.imshow('lane',final_frame)
    		
        
    		if cv2.waitKey(25) & 0xFF == ord('q'):
        		 cv2.destroyAllWindows()
        		 break              
        
        
        
        
        

    def ObjectDetection(self, InputImage):
        ImageHeight, ImageWidth, _ = InputImage.shape
        InputImage = cv2.resize(InputImage, (self.ImageSize, self.ImageSize))
        InputImage = cv2.cvtColor(InputImage, cv2.COLOR_BGR2RGB).astype(numpy.float32)
        InputImage = InputImage / 255.0 * 2.0 - 1.0
        InputImage = numpy.reshape(InputImage, [1, self.ImageSize, self.ImageSize, 3])

        OutputImage = self.sess.run(self.yolo.logits, feed_dict = {self.yolo.images: InputImage})

        results = self.CalculateOutput(OutputImage)

        for i in range(len(results)):
            results[i][1] *= (1.0 * ImageWidth / self.ImageSize)
            results[i][2] *= (1.0 * ImageHeight / self.ImageSize)
            results[i][3] *= (1.0 * ImageWidth / self.ImageSize)
            results[i][4] *= (1.0 * ImageHeight / self.ImageSize)

        return results









    def CalculateOutput(self, output):
        output = numpy.reshape(output, [self.CellSize, self.CellSize, self.BoxesPerCell, 5 + self.num_classes])
        AnchorBox = numpy.reshape(output[:, :, :, :4], [self.CellSize, self.CellSize, self.BoxesPerCell, 4])    
        AnchorBox = self.BoundingBoxes(AnchorBox) * self.ImageSize

        confidence = numpy.reshape(output[:, :, :, 4], [self.CellSize, self.CellSize, self.BoxesPerCell])    
        confidence = 1.0 / (1.0 + numpy.exp(-1.0 * confidence))
        confidence = numpy.tile(numpy.expand_dims(confidence, 3), (1, 1, 1, self.num_classes))

        classes = numpy.reshape(output[:, :, :, 5:], [self.CellSize, self.CellSize, self.BoxesPerCell, self.num_classes])    #classes
        classes = numpy.exp(classes) / numpy.tile(numpy.expand_dims(numpy.sum(numpy.exp(classes), axis=3), axis=3), (1, 1, 1, self.num_classes))

        Propability = classes * confidence

        FilteredPropabilities = numpy.array(Propability >= self.Threshhold, dtype = 'bool')
        FilteredIndex = numpy.nonzero(FilteredPropabilities)
        FilterBox = AnchorBox[FilteredIndex[0], FilteredIndex[1], FilteredIndex[2]]
        PropabilityFilter = Propability[FilteredPropabilities]
        ClassIndexNumber = numpy.argmax(FilteredPropabilities, axis = 3)[FilteredIndex[0], FilteredIndex[1], FilteredIndex[2]]

        sort_num = numpy.array(numpy.argsort(PropabilityFilter))[::-1]
        FilterBox = FilterBox[sort_num]
        PropabilityFilter = PropabilityFilter[sort_num]
        ClassIndexNumber = ClassIndexNumber[sort_num]

        for i in range(len(PropabilityFilter)):
            if PropabilityFilter[i] == 0:
                continue
            for j in range(i+1, len(PropabilityFilter)):
                if self.IntersectionOverUnion(FilterBox[i], FilterBox[j]) > 0.5:
                    PropabilityFilter[j] = 0.0

        FilteredPropabilities = numpy.array(PropabilityFilter > 0, dtype = 'bool')
        PropabilityFilter = PropabilityFilter[FilteredPropabilities]
        FilterBox = FilterBox[FilteredPropabilities]
        ClassIndexNumber = ClassIndexNumber[FilteredPropabilities]

        results = []
        for i in range(len(PropabilityFilter)):
            results.append([self.classes[ClassIndexNumber[i]], FilterBox[i][0], FilterBox[i][1],
                            FilterBox[i][2], FilterBox[i][3], PropabilityFilter[i]])

        return results










    def BoundingBoxes(self, InputBoxes):
        Ofset = numpy.transpose(numpy.reshape(numpy.array([numpy.arange(self.CellSize)] * self.CellSize * self.BoxesPerCell),[self.BoxesPerCell, self.CellSize, self.CellSize]), (1, 2, 0))
        boundingBox = numpy.stack([(1.0 / (1.0 + numpy.exp(-1.0 * InputBoxes[:, :, :, 0])) + Ofset) / self.CellSize,(1.0 / (1.0 + numpy.exp(-1.0 * InputBoxes[:, :, :, 1])) + numpy.transpose(Ofset, (1, 0, 2))) / self.CellSize, numpy.exp(InputBoxes[:, :, :, 2]) * numpy.reshape(self.Anchor[:5], [1, 1, 5]) / self.CellSize,numpy.exp(InputBoxes[:, :, :, 3]) * numpy.reshape(self.Anchor[5:], [1, 1, 5]) / self.CellSize])
        return numpy.transpose(boundingBox, (1, 2, 3, 0))






    def IntersectionOverUnion(self, FirstBox, SecondBox):
        Width = min(FirstBox[0] + 0.5 * FirstBox[2], SecondBox[0] + 0.5 * SecondBox[2]) - max(FirstBox[0] - 0.5 * FirstBox[2], SecondBox[0] - 0.5 * SecondBox[2])
        Height = min(FirstBox[1] + 0.5 * FirstBox[3], SecondBox[1] + 0.5 * SecondBox[3]) - max(FirstBox[1] - 0.5 * FirstBox[3], SecondBox[1] - 0.5 * SecondBox[3])
        if Width <= 0 or Height <= 0:
            Intersection = 0
        else:
            Intersection = Width * Height

        return Intersection / (FirstBox[2] * FirstBox[3] + SecondBox[2] * SecondBox[3] - Intersection)





    

    def draw(self, InputImage, result):
        ImageHeight, ImageWidth, _ = InputImage.shape
        for i in range(len(result)):
            MinimumX = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            MaximumX = min(int(result[i][1] + 0.5 * result[i][3]), ImageWidth)
            MinimumY = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            MaximumY = min(int(result[i][2] + 0.5 * result[i][4]), ImageHeight)
            Xmiddlepoint = MinimumX
            Xmiddlepoint = math.floor(Xmiddlepoint)
            Ymiddlepoint = (MinimumY+MaximumY)/2
            Ymiddlepoint = math.floor(Ymiddlepoint)
            self.DrawTraceLines(Xmiddlepoint,Ymiddlepoint,InputImage)
            cv2.rectangle(InputImage, (MinimumX, MinimumY), (MaximumX, MaximumY), 200, 1)
            cv2.putText(InputImage, result[i][0] + ':%.2f' % result[i][5], (MinimumX + 1, MinimumY + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, 200, 1)
            print(result[i][0], ':%.2f%%' % (result[i][5] * 100 ))
            
            
                  
            
    def DrawTraceLines(self,point_x,point_y,image):
    	
    	point_list_x.append(point_x)
    	point_list_y.append(point_y)
    	was_connected.append(False)
    	for i in range(len(point_list_x)):
    		print(point_list_x[i],point_list_y[i])
    		cv2.line(image,(point_list_x[i],point_list_y[i]),(point_list_x[i]+1,point_list_y[i]+1),200,1)
    		for j in range(len(point_list_y)):
    			if (point_list_x[j]-point_list_x[i]<=50 and point_list_y[j]-point_list_y[i]<=10 and was_connected[i]==False):
    				cv2.line(image,(point_list_x[i],point_list_y[i]),(point_list_x[j],point_list_y[j]),200,5)
    				was_connected[i]=True
    				was_connected[j]=True
    	


      
                
    def MakeCoordinates(self,frame,lines):
        slope,intercept = lines
        print(frame.shape)
        y1 = frame.shape[0]
        y2 = int(y1*3/5)
        x1 = int(y1 - intercept/slope)
        x2 = int(y2 - intercept/slope) 
        return numpy.array([x1,y1,x2,y2])
        
        
        
        
    def AverageSlope(self,frame,lines):
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
        LeftLine=self.MakeCoordinates(frame,AverageLeftSide)
        RightLine=self.MakeCoordinates(frame,AverageRightSide)
        return numpy.array([LeftLine,RightLine])         
    
    
    def DrawLines(self,frame,lines):
        ImageLines = numpy.zeros_like(frame)
        if lines is not None: 
             for line in lines:
                 x1,y1,x2,y2 = line.reshape(4)
                 cv2.line(ImageLines,(x1,y1),(x2,y2),(255,0,0),5)
        return ImageLines                
    
    def single_line(self,frame,lines):
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
        PolyLeft = numpy.poly1d(numpy.polyfit(
        LeftLineY,
        LeftLineX,
        deg=1
        ))
        StartLeftX = int(PolyLeft(MaxuimumY))
        EndLeftX = int(PolyLeft(MinimumY))
        poly_right = numpy.poly1d(numpy.polyfit(
            RightLineY,
            RightLineX,
            deg=1
        ))
        StartRightX = int(poly_right(MaxuimumY))
        EndRightX = int(poly_right(MinimumY))
        ImageLines = self.DrawLines(
        image,
        [[
            [StartLeftX, MaxuimumY, EndLeftX, MinimumY],
            [StartRightX, MaxuimumY, EndRightX, MinimumY],
        ]],
        thickness=5,
        )
        return ImageLines
        
        
        
    
    def FrameProcessing(self,frame):
        gray =cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        blur =cv2.GaussianBlur(gray,(5,5),0)
        canny=cv2.Canny(blur,100,200)    
        return canny
                    
    
    def RegionOfIntrest(self,frame):
        triangle = numpy.array([[(94,720),(1031,720),(781,349),(362,353)]])
        
        mask = numpy.zeros_like(frame)
        cv2.fillPoly(mask,triangle,(255,255,255))
        masked_frame = cv2.bitwise_and(frame,mask)
        return masked_frame
        
    

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt', type = str)   
    parser.add_argument('--weight_dir', default = 'output', type = str)
    parser.add_argument('--data_dir', default = 'data', type = str)
    parser.add_argument('--gpu', default = '', type = str)    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu   
    weights_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    yolo = yolo_v2(False)   

    detector = Detector(yolo, weights_file)

    
    detector.ScreenDetect()

if __name__ == '__main__':
    main()
