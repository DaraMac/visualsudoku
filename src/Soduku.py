import cv2
import numpy as np

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
            if area > max_area & len(approx) == 4:
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
    print("original points: ")
    print(points)
    print("new points: ")
    print(new_points)
    return new_points

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


img_path = 'input/99.jpg'
img_h = 550
img_w = 550
find_sudoku = False

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
    find_sudoku == True
    biggest = reframe(biggest)
    cv2.drawContours(img_biggest_contour, biggest, -1, (0, 255, 0), 15) 
    #prepare ponits before warp
    pts_biggest_contour = np.float32(biggest) 
    pts_img = np.float32([[0, 0],[img_w, 0], [0, img_h],[img_w, img_h]])
    #Perspective Transform
    matrix = cv2.getPerspectiveTransform(pts_biggest_contour, pts_img) 
    img_warp = cv2.cvtColor(cv2.warpPerspective(img, matrix, (img_h, img_w)),cv2.COLOR_BGR2GRAY)
else:
	#raise Exception("Could not find Sudoku puzzle outline.")
    print("Could not find Sudoku puzzle outline.")

if find_sudoku:
    steps_demo = ([img,image_threshold, img_contours, img_warp])
else:
    steps_demo = ([img,image_threshold, img_contours])
    
img_pipeline = img_stack(steps_demo, 1)
cv2.imshow('Images', img_pipeline)
cv2.waitKey(0)
cv2.destroyAllWindows()
