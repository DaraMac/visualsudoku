# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:07:24 2021

@author: My Laptop
"""

import cv2
import imutils
import numpy as np
sudoku_v_cells = 9
sudoku_h_cells = 9
# Preprocess image function:
# Extract edges and use morphology operators to post process them to retain
# only vertical and horizontal edges
def image_preprocessing(image):
    line_min_length = 10
    image_blur = cv2.GaussianBlur(image, (1, 1), cv2.BORDER_DEFAULT)
    image_edges = cv2.Canny(image=image_blur, threshold1=50,
                                threshold2=100)

    kernel = np.ones((3, 3), 'uint8')
    # use the close operator to fill the gaps between edges of thik lines
    image_edges = cv2.morphologyEx(image_edges, cv2.MORPH_CLOSE, kernel)

    # remove all edges where are neither vertical nor horizontal
    horizontal_kernal = np.ones((1, line_min_length), np.uint8)
    vertical_kernal = np.ones((line_min_length, 1), np.uint8)
    img_bin_horizontal = cv2.morphologyEx(image_edges, cv2.MORPH_OPEN,
                                          horizontal_kernal)
    img_bin_vertical = cv2.morphologyEx(image_edges, cv2.MORPH_OPEN,
                                        vertical_kernal)
    image_horizontal_vertical_edges = img_bin_horizontal | img_bin_vertical

    # use thinning operator to thin the edges
    thin = np.zeros(image_horizontal_vertical_edges.shape, dtype='uint8')
    erode = cv2.erode(image_horizontal_vertical_edges, kernel)
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    subset = erode - opening
    image_horizontal_vertical_edges = cv2.bitwise_or(subset, thin)

    return image_horizontal_vertical_edges

# Find all contours function:
# We entered a threshold image and the function find all contours and return them
def find_all_contours(image_threshold):
    contours, _ = cv2.findContours(image_threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)
    rectangle_contours = []
    for contour in contours:
        rectangle_contours.append(cv2.boundingRect(contour))
    print('num of contours: ', len(rectangle_contours))
    return rectangle_contours


# find only horizontal and vertical lines
def find_lines(image_threshold, original_image_min_side):
    threshold = original_image_min_side // 50
    minLineLength = original_image_min_side // 40
    maxLineGap = original_image_min_side // 30
    lines= cv2.HoughLinesP(image=image_threshold, rho=10, theta=np.pi/2,
                           threshold=threshold, minLineLength=minLineLength,
                           maxLineGap=maxLineGap)
    print('num of lines: ', len(lines))
    return lines

def point_is_inside(x, y, rectangle):
    eps = 10
    if (x >= (rectangle[0] - eps) and
        x <= (rectangle[0] + rectangle[2] + eps) and
        y >= (rectangle[1] - eps) and
        y <= (rectangle[1] + rectangle[3] + eps)):
        return True
    return False

def is_inside(line, rectangle):
    return (point_is_inside(line[0][0], line[0][1], rectangle) and
            point_is_inside(line[0][2], line[0][3], rectangle))

def has_appropriate_length(line, rectangle, threshold):
    if line[0][0] == line[0][2]:  # vertical line
        return abs(line[0][1] - line[0][3]) <= (rectangle[3] * threshold)
    else:
        return abs(line[0][0] - line[0][2]) <= (rectangle[2] * threshold)

# use voting mechanism to find sudoku table. If a line is inside a contour and
# its length is larger than a threshold it vote positive for the given contour
# otherwise it votes negative
def find_sudoku(contours, lines):
    votes_per_contour = [0] * len(contours)
    for index, contour in enumerate(contours):
        for line in lines:
            if is_inside(line, contour) and has_appropriate_length(line, contour, 0.25):
                votes_per_contour[index] += 1
            else:
                votes_per_contour[index] -= 1

    max_vote = max(votes_per_contour)
    contour_index = votes_per_contour.index(max_vote)
    return contours[contour_index]

# Reorder points
# The smallest value is the origin, the biggest value is the height and weight
# Take difference of above, the opsitive and negative value are corresspoding opints that connet to the origin
def reorder(points):
    print('points',len(points))
    points = points.reshape((4, 2))
    print(len(points))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    # print("original points: ")
    # print(points)
    # print("new points: ")
    # print(new_points)
    return new_points

# Stack images
# Use for demo only
def img_stack(img_array,scale):
    rows = len(img_array)
    cols = len(img_array[0])
    rowsAvailable = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y]= cv2.cvtColor( img_array[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
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


img_path = '98.jpg'

original_image = cv2.imread(img_path)
img = original_image
# img = cv2.resize(img, (img_w, img_h))
original_image_height = img.shape[0]
original_image_width = img.shape[1]
img_w = original_image_width
img_h = original_image_height
original_image_min_side = min(original_image_height, original_image_width)
image_threshold = image_preprocessing(img)
lines = find_lines(image_threshold, original_image_min_side)
#demo use only
img_pipeline = np.zeros((img_w, img_h, 3), np.uint8)
img_contours = img.copy()
img_lines = img.copy()
img_sudoku_contour = img.copy()
contours = find_all_contours(image_threshold)
sudoku_contour = find_sudoku(contours, lines)

for line in lines:
    img_lines = cv2.line(img_lines, (line[0][0], line[0][1]),
                         (line[0][2], line[0][3]), (0, 255, 0), 2)

for box in contours:
    box = [int(p) for p in box]
    img_contours = cv2.rectangle(img_contours, (box[0], box[1]), (box[0]+box[2],
                                 box[1]+box[3]), (0, 0, 255), 2)

img_sudoku_contour = cv2.rectangle(img_sudoku_contour,
                                  (sudoku_contour[0], sudoku_contour[1]),
                                  (sudoku_contour[0]+sudoku_contour[2],
                                   sudoku_contour[1]+sudoku_contour[3]),
                                  (255, 0, 0), 2)

steps_demo = ([img, image_threshold, img_lines, img_contours, img_sudoku_contour])
img_pipeline = img_stack(steps_demo, 0.5)
cv2.imshow('Images', img_pipeline)
cv2.imwrite('sudoku_contour.jpg', img_sudoku_contour)
cv2.imwrite('pipeline.jpg', img_pipeline)

cv2.waitKey(0)
cv2.destroyAllWindows()
