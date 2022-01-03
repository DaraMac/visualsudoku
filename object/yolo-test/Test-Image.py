# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:02:47 2021

@author: Solmaz
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

## provide the path for testing cofing file and tained model form colab
net = cv2.dnn.readNetFromDarknet("yolov3_custom_test.cfg",r"C:\Users\Solmaz\Documents\Group Project\Object_Detection\YOLO\Test\yolov3_custom_final.weights")

### Change here for custom classes for trained model 

classes = ['sudoku']

img = cv2.imread(r'C:\Users\Solmaz\Documents\Group Project\Object_Detection\YOLO\Test\Test_Images\img126.jpg')
#img = cv2.resize(img,(416,416))
#cv2.imshow('img1',img)
hight,width,_ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

net.setInput(blob)

output_layers_name = net.getUnconnectedOutLayersNames()

layerOutputs = net.forward(output_layers_name)
#print(len(layerOutputs))
boxes =[]
confidences = []
class_ids = []

for output in layerOutputs:
   for detection in output:
       score = detection[5:]
       class_id = np.argmax(score)
       confidence = score[class_id]
       
       if confidence > 0.5:
          #print(confidence)
          center_x = int(detection[0] * width)
          center_y = int(detection[1] * hight)
          w = int(detection[2] * width)
          h = int(detection[3]* hight)
          x = int(center_x - w/2)
          y = int(center_y - h/2)
          boxes.append([x,y,w,h])
        
          confidences.append((float(confidence)))
          
          class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size =(len(boxes),4))
if  len(indexes)>0:
    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        #print(x,y,w,h)
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label + " " + confidence, (x+10,y-10),font,1.5,color,2)

cv2.imshow('img',img)
cv2.waitKey(0)  
cv2.destroyAllWindows() 
