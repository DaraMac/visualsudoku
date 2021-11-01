import cv2
import numpy as np
import math
from scipy import ndimage
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model


sudoku_v_cells = 9
sudoku_h_cells = 9


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

def displayNumbers(img,numbers,color = (255, 255, 255)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

# Preprocess image function:
# Convert the image to grayscale
# Add some blur for easier detection
# Apply adaptive threshold
def image_preprocessing(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 2)
    image_threshold = cv2.adaptiveThreshold(image_blur, 255, 1, 1, 11, 2)
    return image_threshold

# Pre-processing cells
# Convert to prepare format to recognize (the images ready for detection model)
def cell_preprocessing(cell):
    cell = ~cell
    new_cells = np.asarray(cell)
    new_cells = new_cells[4:img.shape[0] - 4, 4:new_cells.shape[1] -4]
    new_cells = cv2.resize(new_cells, (28, 28))
    new_cells = tf.keras.utils.normalize(new_cells, axis = 1)
    new_cells = new_cells.reshape(1, 28, 28, 1)
    return new_cells

def cell_preprocessing2(cell):
    cell = ~cell
    resized = cv2.resize(cell, (28, 28), interpolation =cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis = 1)
    normalize_newimg = np.array(newimg).reshape(-1, 28, 28, 1)
    return normalize_newimg

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

# Get prediction and save result
def predect_digits(cells,model):
    result = []
    for cell in cells:
        new_cells = cell_preprocessing2(cell)
        predictions = model.predict(new_cells)
        # index = model.predict_classes(new_cells) # replaced this with line below as it is deprecated
        index = np.argmax(model.predict(new_cells), axis=-1) # this also gives invalid value in double_scalars
        probability_value = np.amax(predictions, axis = -1)
        #print(index, probability_value)
        if probability_value > 0.8:
            result.append(index[0])
        else:
            result.append(0)
    return result


# This function is used for separating the digit from noise in each box of the boxes
# The sudoku board will be cropped into 9x9 small square image in the split boxes function
# each of those box is a cropped image
def digit_component(image):
    image = image.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) <= 1:
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    output_image = np.zeros(output.shape)
    output_image.fill(255)
    output_image[output == max_label] = 0
    return output_image


def prepossessing_for_model(main_board):
    main_board = cv2.GaussianBlur(main_board, (5, 5), 2)
    main_board = cv2.adaptiveThreshold(main_board, 255, 1, 1, 11, 2)
    main_board = cv2.bitwise_not(main_board)
    _, main_board = cv2.threshold(main_board, 10, 255, cv2.THRESH_BINARY)
    return main_board


# Calculate how to centralize the image using its center of mass
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty


# Shift the image using what get best shift returns
def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


# First we init an empty cells to store the sudoku board digits
# Remove all boundaries on the edges of each image
# Then clear the digits images and remove noise from it
# Apply binary threshold to get ready for recognition model
def crop_cell2(main_board):
    # Init empty 9 by 9 grid

    cells = []
    # Calculate the width and height to split main board to 81 image
    height = main_board.shape[0] // sudoku_v_cells
    width = main_board.shape[1] // sudoku_h_cells

    # Offset is used to get rid of the boundaries
    offset_width = math.floor(width / 10)
    offset_height = math.floor(height / 10)

    # Split the sudoku board into 9x9 squares ( 81 images )
    for i in range(sudoku_h_cells):
        for j in range(sudoku_v_cells):

            # Crop with offset ( we don't want to include the boundaries )
            crop_image = main_board[height * i + offset_height:height * (i + 1) - offset_height, width * j + offset_width:width * (j + 1) - offset_width]
            
            # But after that it will still have some boundary lines left
            # So we remove all black lines near the edges of each image
            # The ratio = 0.6 means if 60% of the pixels are black then remove this boundaries
            ratio = 0.6

            # Top
            while np.sum(crop_image[0]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]

            # Bottom
            while np.sum(crop_image[:, -1]) <= (1 - ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)

            # Left
            while np.sum(crop_image[:, 0]) <= (1 - ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)

            # Right
            while np.sum(crop_image[-1]) <= (1 - ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]

            # Take the largest connected component (the digit) and remove all noises from it
            crop_image = cv2.bitwise_not(crop_image)
            crop_image = digit_component(crop_image)

            # Resize each image to prepare it to the model

            crop_image = cv2.resize(crop_image, (28, 28))
            
            # Detecting white cell if there is a huge white area in the center of the image
            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]

            # Apply binary threshold to make digits more clear
            _, crop_image = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY)
            crop_image = crop_image.astype(np.uint8)

            # Centralize the image according to center of mass
            crop_image = cv2.bitwise_not(crop_image)

            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image, shift_x, shift_y)
            crop_image = shifted


            crop_image = cv2.bitwise_not(crop_image)
            cells.append(crop_image)
            
    return cells




img_path = 'input/3.jpg'
model_path ='model/model.h5'
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
    
    main_board = prepossessing_for_model(img_warp)
    #main_board = img_warp
    model = load_model(model_path)
    
    cells = crop_cell(main_board)
    cells = crop_cell2(main_board)

    #cv2.imwrite('numbers/savedImage2.jpg', cells[1], [cv2.IMWRITE_JPEG_QUALITY, 100])

    digits = predect_digits(cells, model)
    print(digits)

    imgDetectedDigits = img_pipeline.copy()
    imgDetectedDigits = displayNumbers(imgDetectedDigits, digits, color=(255, 255, 255))
 
steps_demo = ([img, img_warp,main_board, imgDetectedDigits])
steps_demo2 = ([cells])
img_pipeline = img_stack(steps_demo, 1)
img_pipeline2 = img_stack(steps_demo2, 1)
cv2.imshow('Images', img_pipeline)
cv2.imshow("Cells",img_pipeline2)  
cv2.waitKey(0)
cv2.destroyAllWindows()
