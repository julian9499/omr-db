import math

import cv2 as cv2
import numpy as np


COLUMNS = 9
ROWS = 30
ANSWERS = 3

epsilon = 10 #image error sensitivity
filename = "temp/sanitycheck.jpg"

# load tracking tags
tags = [cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)]

scaling = [869.0, 840.0] #scaling factor for 8.5in. x 11in. paper
columns = [[55.5 / scaling[0], 66.5 / scaling[1]]] #dimensions of the columns of bubbles
colspace = 76.8 /scaling[0]
radius = 6.5 / scaling[0] #radius of the bubbles
spacing = [24.9 / scaling[0], 19.9 / scaling[1]] #spacing of the rows and columns

# Load the image from file
img = cv2.imread(filename)
height, width, channels = img.shape


corners = []  # array to hold found corners


def FindCorners(paper, drawRect):
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY) #convert image of paper to grayscale

    #scaling factor used later
    ratio = height / width

    #error detection
    if ratio == 0:
        return -1

    #try to find the tags via convolving the image
    for tag in tags:
        tag = cv2.resize(tag, (0,0), fx=ratio, fy=ratio) #resize tags to the ratio of the image

        #convolve the image
        convimg = (cv2.filter2D(np.float32(cv2.bitwise_not(gray_paper)), -1, np.float32(cv2.bitwise_not(tag))))

        #find the maximum of the convolution
        corner = np.unravel_index(convimg.argmax(), convimg.shape)

        #append the coordinates of the corner
        corners.append([corner[1], corner[0]]) #reversed because array order is different than image coordinate

    #draw the rectangle around the detected markers
    if drawRect:
        for corner in corners:
            cv2.rectangle(paper, (corner[0] - int(ratio * 25), corner[1] - int(ratio * 25)),
            (corner[0] + int(ratio * 25), corner[1] + int(ratio * 25)), (0, 255, 0), thickness=2, lineType=8, shift=0)

    #check if detected markers form roughly parallel lines when connected
    if corners[0][0] - corners[2][0] > epsilon:
        return None

    if corners[1][0] - corners[3][0] > epsilon:
        return None

    if corners[0][1] - corners[1][1] > epsilon:
        return None

    if corners[2][1] - corners[3][1] > epsilon:
        return None

    return

FindCorners(img, False)
print(corners)

desired_points = np.float32([[68, 424], [1489, 424], [68, 1797], [1489, 1797]])
points = np.float32(corners)

M = cv2.getPerspectiveTransform(points, desired_points)
sheet = cv2.warpPerspective(img, M, (1589, 1997))


img = sheet
height, width, channels = img.shape
corners = []
FindCorners(img, True)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to binarize it
treshImg, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

xdis, ydis = corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]
answersBoundingbox = [(int(corners[0][0] + 0.035 *xdis), corners[0][1] + int(0.055 *ydis)), (corners[3][0] - int(0.15 * xdis), corners[3][1] - int(0.21 * ydis))]
cv2.rectangle(img, answersBoundingbox[0],
        answersBoundingbox[1], (0, 255, 0), thickness=2, lineType=8, shift=0)

# calculate dimensions for scaling
dimensions = [corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]]


boulders = list()
for i in range(0, ROWS):
    boulders.append([0,0,0])

# iterate over test questions
for i in range(0, ROWS):  # rows
    for k in range(0, COLUMNS):  # columns
        for j in range(0, ANSWERS):  # answers
            # coordinates of the answer bubble
            x1 = int((columns[0][0] + colspace * k + j * spacing[0] - radius) * dimensions[0] + corners[0][0])
            y1 = int((columns[0][1] + i * spacing[1] - radius) * dimensions[1] + corners[0][1])
            x2 = int((columns[0][0] + colspace * k + j * spacing[0] + radius) * dimensions[0] + corners[0][0])
            y2 = int((columns[0][1] + i * spacing[1] + radius) * dimensions[1] + corners[0][1])

            # draw rectangles around bubbles
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1, lineType=8, shift=0)

            roi = thresh[y1:y2, x1:x2]

            percentile = (np.sum(roi == 255)/((y2-y1) * (x2-x1))) * 100
            print(percentile)

            if percentile > 40.0:
                if (j != 0 and boulders[i][j] == 0) or j == 0:
                    boulders[i][j] = k + 1

for i in range(0, ROWS):
    x1 = int((columns[0][0] + colspace * 9.7) * dimensions[0] + corners[0][0])
    y1 = int((columns[0][1]+0.005 + i * spacing[1]) * dimensions[1] + corners[0][1])
    cv2.putText(img, str(boulders[i][1]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 0), 2)

    x2 = int((columns[0][0] + colspace * 10.5) * dimensions[0] + corners[0][0])
    cv2.putText(img, str(boulders[i][2]), (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 0), 2)


print(boulders)
# Show the image with the rectangles
cv2.imwrite(f"checkboxes_{filename}", img)
cv2.imshow("Checkboxes", cv2.resize(img, (0, 0), fx=0.7, fy=0.7))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute the average height of all the bounding boxes
# avg_height = sum([h for (x, y, w, h) in bounding_boxes if
#                   w > width / 18.5 and h > height / 54 and 3.0 * h > w > 2.0 * h and y > 200]) / len(bounding_boxes)

# Compute the rotation angle
# angle = np.arctan(avg_height / img.shape[1])
#
# # # Rotate the image
# # rows, cols = img.shape[:2]
# # rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle * 180 / np.pi, 1)
# # img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Threshold the image to binarize it
# treshImg, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#
# # Find contours in the thresholded image
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# boxes = {}
# lines = set()
# for i in range(0, 33):
#     boxes[i] = list()
#
# # Loop over the contours
# for contour in contours:
#     # Get the bounding box of the  contour
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # Check if the contour is a checkbox
#     if 3 * h > w > 2 * h and w * h > width * height / 1000 and y > 150:
#         collision = True
#
#         # Crop the region of interest (ROI)
#         roi = gray[y:y + h, x:x + w]
#         d = 44.3
#         line = int(round((y - 222) / (height / d))) + 1
#         lines.add(line)
#         box = {}
#         box["boundingBox"] = (x, y, w, h)
#
#         # Count the number of white pixels in the ROI
#         white_pixels = np.sum(roi == 255)
#         # print(y)
#         # edges = cv2.Canny(roi, 100, 200)
#
#         # Draw a rectangle around the checkbox
#         # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         cv2.rectangle(img, (60, y), (110, y + h - 5), (255, 255, 255), -1)
#         cv2.putText(img, f"{line}", (60, y + int((height / 45.1) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#         # cv2.imshow("Edges", edges)
#         # cv2.waitKey(0)
#         # Use Tesseract to extract the text from the ROI
#         # text = pytesseract.image_to_string(roi)
#
#         # Write the text on the image
#         # cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         # Check if the ROI is filled or not filled
#         if not collision:  # white_pixels > (w * h) * 0.8:
#             # print("Checkbox is not filled")
#             cv2.putText(img, f"{line}", (x, int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#             box["filled"] = False
#         else:
#             # print("Checkbox is filled")
#             cv2.putText(img, f"{line}", (x, int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
#             box["filled"] = True
#
#         boxes[line].append(box)
