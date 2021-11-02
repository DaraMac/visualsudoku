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



def change_gray(img):
    # img = cv2.imread(img_path)
    width, height = img.shape[:2][::-1]
    img_resize = cv2.resize(img, (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_CUBIC)
    print("img_reisze shape:{}".format(np.shape(img_resize)))

    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
    print("img_gray shape:{}".format(np.shape(img_gray)))
    
    return img_gray

def accessPiexl(img):
    # change_gray(img)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img


def accessBinary(img, threshold=170):
    img = accessPiexl(img)

    kernel = np.ones((3, 3), np.uint8)
    # img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.medianBlur(img, 3)


    img = cv2.erode(img, kernel, iterations=1)

    # _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -9)

    img = cv2.dilate(img, kernel, iterations=1)
    return img

def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        #enumerate() 
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints


def findBorderHistogram(path):
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)                     

    hori_vals = np.sum(img, axis=1)

    hori_points = extractPeek(hori_vals)

    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=10)
       
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            
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


def transMNIST(path, borders, size=(28, 28)):
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        h = abs(border[0][1] - border[1][1])#
        # w = abs(border[0][0] - border[1][0])#

        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        h_extend = h // 5
        # print(h_extend)
        targetImg = cv2.copyMakeBorder(borderImg, h_extend, h_extend, int(extendPiexl*1.1), int(extendPiexl*1.1), cv2.BORDER_CONSTANT)
        # targetImg = cv2.copyMakeBorder(borderImg, 20, 20, h_extend, h_extend,cv2.BORDER_CONSTANT)
        # targetImg = cv2.resize(borderImg, size)

        # if w < (h//3):
        #     targetImg = cv2.copyMakeBorder(targetImg, 4, 4, 20, 20, cv2.BORDER_CONSTANT
        #     print("1")
        # else:
        #     targetImg = cv2.copyMakeBorder(targetImg, 4, 4, 6, 6, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        # cv2.imshow('test', targetImg)
        # cv2.waitKey(0)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData




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
