import math
import cv2 as cv2
import numpy as np


COLUMNS = 5
ROWS = 30
ANSWERS = 3

epsilon = 10 #image error sensitivity
filename = "./scanned/form11.jpg"

# load tracking tags
tags = [cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)]

scaling = [869.0, 840.0] #scaling factor for 8.5in. x 11in. paper
columns = [[83.0 / scaling[0], 42.5 / scaling[1]]] #dimensions of the columns of bubbles
colspace = 135.8 /scaling[0]
radius = 8.5 / scaling[0] #radius of the bubbles
spacing = [35.2 / scaling[0], 22.25 / scaling[1]] #spacing of the rows and columns

# Load the image from file
img = cv2.imread(filename)
height, width, channels = img.shape


corners = []  # array to hold found corners


def FindCorners(paper, drawRect):
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY) #convert image of paper to grayscale

    #scaling factor used later
    ratio = width / height

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

# FindCorners(img, False)
# print(corners)
#
# desired_points = np.float32([[68, 424], [1489, 424], [68, 1797], [1489, 1797]])
# points = np.float32(corners)
#
# M = cv2.getPerspectiveTransform(points, desired_points)
# sheet = cv2.warpPerspective(img, M, (1589, 1997))
#
# img = sheet
height, width, channels = img.shape
corners = []
FindCorners(img, True)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to binarize it
treshImg, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

xdis, ydis = corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]
answersBoundingbox = [(int(corners[0][0] + 0.035 *xdis), corners[0][1] + int(0.055 *ydis)), (corners[3][0] - int(0.15 * xdis), corners[3][1] - int(0.21 * ydis))]
# cv2.rectangle(img, answersBoundingbox[0],
#         answersBoundingbox[1], (0, 255, 0), thickness=2, lineType=8, shift=0)

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

            if percentile > 30.0:
                if (j != 0 and boulders[i][j] == 0) or j == 0:
                    boulders[i][j] = k+1

for i in range(0, ROWS):
    x1 = int((columns[0][0] + colspace * 5.1) * dimensions[0] + corners[0][0])
    y1 = int((columns[0][1]+0.005 + i * spacing[1]) * dimensions[1] + corners[0][1])
    cv2.putText(img, str(boulders[i][1]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 0), 2)

    x2 = int((columns[0][0] + colspace * 5.7) * dimensions[0] + corners[0][0])
    cv2.putText(img, str(boulders[i][2]), (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 0), 2)


print(boulders)


# Show the image with the rectangles
cv2.imshow("results", cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()

