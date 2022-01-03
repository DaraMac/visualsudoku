# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 19:36:27 2022


"""

import cv2
import imutils
import numpy as np

sudoku_v_cells = 9
sudoku_h_cells = 9

# Preprocess image function:
# Convert the image to grayscale
# Add some blur for easier detection
# Apply adaptive threshold
def image_preprocessing(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 2)
    image_threshold = cv2.adaptiveThreshold(image_blur, 255, 1, 1, 11, 2)
    return image_threshold

# Find all contours function:
# We entered a threshold image and the function find all contours and return it
# Optional cv2.RETR_TREE
def find_all_contours(image_threshold):
    contours, _ = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Find biggest contour function:
# Extract the biggest contours because the sudoku board is the biggest one
# We loop in all contours
# Then check the area of each one because the small contours it's maybe noise or other unimportant details
# Finally get only biggest shapes points
def biggest_contour(contours):
    max_area = 0
    biggest_contour = np.array([])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                max_area = area
                biggest_contour = approx
    return biggest_contour,max_area

# Reorder points 
# The smallest value is the origin, the biggest value is the height and weight
# Take difference of above, the opsitive and negative value are corresspoding opints that connet to the origin
def reframe(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    print("original points: \n", points)
    print("new points: \n", new_points)
    return new_points

# Split sudoku into sudoku_v_cells * sudoku_h_cells cells
def crop_cell(img):
    rows = np.vsplit(img,sudoku_v_cells)
    cells=[]
    for r in rows:
        cols= np.hsplit(r,sudoku_h_cells)
        for cell in cols:
            cells.append(cell)
    print("number of cells: ", len(cells))
    return cells

# Stack images
# Use for demo only
def img_stack(img_array,scale):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range ( 0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y]= cv2.cvtColor( img_array[x][y], cv2.COLOR_GRAY2BGR)
        img_pipeline = np.zeros((height, width, 3), np.uint8)
        hor = [img_pipeline]*rows
        hor_con = [img_pipeline]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(img_array)
        hor_con= np.concatenate(img_array)
        ver = hor
    return ver


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


indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.3)

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
crop_image = img[x:w,y:h]
cv2.imwrite("box.jpg" %i, crop_image)

img_path = 'C:\\Users\\Solmaz\\Documents\\Group Project\\Object_Detection\\YOLO\\Test\\box.jpg'
img_h = 540
img_w = 540

img = cv2.imread(img_path)
img = cv2.resize(img, (img_w, img_h))
image_threshold = image_preprocessing(img)
#demo use only
img_pipeline = np.zeros((img_w, img_h, 3), np.uint8) 

img_contours = img.copy() 
img_biggest_contour = img.copy() 
contours = find_all_contours(image_threshold)
cv2.drawContours(img_contours, contours, -1, (155, 0, 155), 3) 

biggest, max_area = biggest_contour(contours) 
if biggest.size != 0:
    biggest = reframe(biggest)
    cv2.drawContours(img_biggest_contour, biggest, -1, (0, 255, 0), 15) 
    #prepare ponits before warp
    pts_biggest_contour = np.float32(biggest) 
    pts_img = np.float32([[0, 0],[img_w, 0], [0, img_h],[img_w, img_h]])
    #Perspective Transform
    matrix = cv2.getPerspectiveTransform(pts_biggest_contour, pts_img) 
    img_warp = cv2.cvtColor(cv2.warpPerspective(img, matrix, (img_h, img_w)),cv2.COLOR_BGR2GRAY)

imgSolvedDigits = img_pipeline.copy()
cells = crop_cell(img_warp)
 
steps_demo = ([img,image_threshold, img_contours, img_warp])
steps_demo2 = ([cells])
img_pipeline = img_stack(steps_demo, 0.5)
img_pipeline2 = img_stack(steps_demo2, 1)
cv2.imshow('Images', img_pipeline)
cv2.imshow("Cells",img_pipeline2)  
cv2.waitKey(0)
cv2.destroyAllWindows()