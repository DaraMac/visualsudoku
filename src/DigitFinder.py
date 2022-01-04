import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

image = cv2.imread('numbers/numbers.jpg')
model_path ='model/model.h5'
model = load_model(model_path)


#Convert to grayscale image
def change_gray(img):
    # img = cv2.imread(img_path)
    # get w and h from input image
    width, height = img.shape[:2][::-1]
    # resize image for better view
    img_resize = cv2.resize(img, (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_CUBIC)
    print("img_reisze shape:{}".format(np.shape(img_resize)))

    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
    print("img_gray shape:{}".format(np.shape(img_gray)))
    
    return img_gray

#Inverted grayscale image, reverse the black and white threshold, process the pixels one by one
def accessPiexl(img):
    # change_gray(img)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

#Inverted binarized image
def accessBinary(img, threshold=170):
    img = accessPiexl(img)

    kernel = np.ones((3, 3), np.uint8)
    # filter
    # img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.medianBlur(img, 3)

    # remove frizz on the edges
    img = cv2.erode(img, kernel, iterations=1)

    # use threshold function for binarization
    # _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -9)

    # edge expansion
    img = cv2.dilate(img, kernel, iterations=1)
    return img

# Scan the rows and columns
# Starting from the row, calculate the sum of the pixel values ​​of each row,
# if they are all black then the sum is 0 (there may be noise, we can set a threshold to filter the noise),
# Lines with fonts are non-zero, we can get the value of line boundary by filtering the boundary according to this picture
# With the above concept, we can scan the rows and columns, e.g. 0, non-zero, non-zero… non-zero, 0, to determine the point of the row and column to find the border of the number

# Find the vertices based on the long vector
def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        #enumerate() 
        #If the white point of the line is greater than the lower limit, and has not appeared, then this is the line starting point.
        if point > min_vals and startPoint == None:
            startPoint = i
        #If the white point of the line is lower than the lower limit, and has appeared, then this is the line end point.
        elif point < min_vals and startPoint != None:
            endPoint = i

        #If both starting point and end point are detected, store them
        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None
            
    #Remove some noise
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints

# Find the edge, return the upper left corner and the lower right corner of the frame (using the histogram to find the edge algorithm (line alignment required))
def findBorderHistogram(path):
    borders = []
    #load as grayscale image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #Inverted binarized image
    img = accessBinary(img)                     
    #Line scan
    hori_vals = np.sum(img, axis=1)


    hori_points = extractPeek(hori_vals)
    
    # for each row, scan column
    # extra determined row
    for hori_point in hori_points:
        
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=10)
       
        #add border
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
                        
            #vect_point[0]the left border of a digit，vect_point[1]the right border of a digit
            # hori_point[0]the upper border of a digit，hori_point[1]the bottom border of a digit
            borders.append(border)
    return borders



def showResults(path, borders, results=None):
    img = cv2.imread(path)

    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))

        if results != None:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            
    cv2.imshow('test', img)
    cv2.waitKey(0)


# for i in range(1, 5):
#     path = r'test' + str(i) + '.png'          
#     borders = findBorderHistogram(path)
#     showResults(path, borders)


def findBorderContours(path, maxArea=100):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            if h > 20:
                border = [(x, y), (x + w, y + h)]
                borders.append(border)
    return borders






for i in range(1, 5):
    path = 'numbers/numbers.jpg'          
    borders = findBorderContours(path)
    # print(borders)
    showResults(path, borders)



image = change_gray(image)
plt.imshow(image)
image = accessPiexl(image)
plt.imshow(image)
image = accessBinary(image)
plt.imshow(image)






# =============================================================================
# grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
# plt.imshow(grey)
# 
# contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 
# preprocessed_digits = []
# for c in contours:
#     x,y,w,h = cv2.boundingRect(c)
#     
#     # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
#     cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
# 
#     # Cropping out the digit from the image corresponding to the current contours in the for loop
#     digit = thresh[y:y+h, x:x+w]
#     
#     # Resizing that digit to (18, 18)
#     resized_digit = cv2.resize(digit, (18,18))
#     
#     # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
#     padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
#     
#     # Adding the preprocessed digit to the list of preprocessed digitsq
#     preprocessed_digits.append(padded_digit)
# print("\n\n\n----------------Contoured Image--------------------")
# plt.imshow(image, cmap="gray")
# plt.show()
#     
# 
# 
# 
# 
# 
# 
# 
# inp = np.array(preprocessed_digits)
# 
# for digit in preprocessed_digits:
#     prediction = model.predict(digit.reshape(1, 28, 28, 1))  
#     
#     print ("\n\n---------------------------------------\n\n")
#     print ("=========PREDICTION============ \n\n")
#     plt.imshow(digit.reshape(28, 28), cmap="gray")
#     plt.show()
#     print("\n\nFinal Output: {}".format(np.argmax(prediction)))
#     
#     print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
#     
#     hard_maxed_prediction = np.zeros(prediction.shape)
#     hard_maxed_prediction[0][np.argmax(prediction)] = 1
#     print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
#     print ("\n\n---------------------------------------\n\n")
# =============================================================================
