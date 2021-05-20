import numpy as np

import cv2
from matplotlib import pyplot as plt
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 # <-- This line altered for grayscale.

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mask_image(image):
    # image processing

    # display the original image converted to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.blur(hsv_image, (40, 40))
    # gamma correction
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.8) * 255.0, 0, 255)
    hsv_image = cv2.LUT(blurred, lookUpTable)

    # define range of blue color in HSV
    lower_blue = np.array([0, 70, 0])
    upper_blue = np.array([50, 150, 255])
    # Threshold the HSV image to get only blue colors
    hsv_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    masked_image = cv2.bitwise_and(image, image, mask=hsv_mask)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


def lane_detection(frame):
    region_of_interest_vertices = [
        (0, frame.shape[0]),
        (frame.shape[1] / 2, frame.shape[0] / 2),
        (frame.shape[1], frame.shape[0]),
    ]
    print(region_of_interest_vertices)
    # cv2.imshow('original',frame)
    # cv2.waitKey()
    # Convert to grayscale here.
    imgconv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # imgconv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    cv2.imshow('gray',imgconv)
    cv2.waitKey()
    # Get the L channel
    # Lchannel = imgconv[:, :, 2]
    # change 250 to lower numbers to include more values as "white"
    # mask = cv2.inRange(Lchannel, 90, 100)
    # apply mask to original image
    # thresholded = cv2.bitwise_and(frame, frame, mask=mask)
    ret, thresholded = cv2.threshold(imgconv,180,255,cv2.THRESH_BINARY)
    cv2.imshow('thresholded',thresholded)
    cv2.waitKey()
    # blur it a little
    # blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(thresholded, 100, 200)
    # Moved the cropping operation to the end of the pipeline.
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 45,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    line_image = draw_lines(frame, lines)  # <---- Add this call.
    plt.imshow(line_image)
    plt.show()
    return line_image

def perspective_correction(image):
    region_of_interest_vertices = [
        (0, image.shape[0]), #bottom left
        (image.shape[1] // 2.1, image.shape[0] / 2), #top left
        (image.shape[1] // 1.9, image.shape[0] / 2), #top right
        (image.shape[1], image.shape[0]), # bottom right
    ]
    print(region_of_interest_vertices)

video = cv2.VideoCapture('Video Nº 11- Ugly Asphalt cracks (S.A. La Beltraneja).mp4')
success, originImg = video.read()
perspective_correction(originImg)