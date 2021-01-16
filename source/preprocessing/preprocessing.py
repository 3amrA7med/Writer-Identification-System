import time

import numpy as np
import cv2
import operator


def extract_hand_written(gray):
    """
    This function returns original cropped image and dilated cropped image 
    for the handwritten part only and exclude any unused parts.
    """
    # Extract grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding using otsu's method to inverse the background and content color.
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply noise removal.
    blur = cv2.blur(thresh_inv, (3, 3))

    # Apply thresholding for better and more general output.
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply dilation to the image to make sure that horizontal lines are continuous.
    # Taking a matrix of size 4,2 as the kernel
    kernel = np.ones((4, 1), np.uint8)
    img_dilation = cv2.dilate(binary, kernel, iterations=3)

    # Find lines by using contour
    # Find contours
    contours, hierarchy = cv2.findContours(img_dilation,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find bounded rectangles surround the contours
    bounding_rectangles = [None] * len(contours)
    for i in range(len(contours)):
        bounding_rectangles[i] = cv2.boundingRect(contours[i])

    # Sort bounding rectangles descendingly according to width
    bounding_rectangles.sort(key=operator.itemgetter(2), reverse=True)

    lines = []  # store the three horizontal lines
    for i in range(len(bounding_rectangles)):
        append = True
        for j in range(len(lines)):
            if abs(bounding_rectangles[i][1] - lines[j][1]) < 100:
                append = False
        if append:
            lines.append(bounding_rectangles[i])
        if len(lines) > 2:
            break

    lines.sort(key=operator.itemgetter(1))  # sort lines ascendingly according to height

    first_crop_line_height = lines[1][1] + lines[1][3]
    second_crop_line_height = lines[2][1]

    # Crop the original image and the dilated image
    segmented_image = img_dilation[first_crop_line_height + 10:second_crop_line_height - 10, :]
    segmented_image_original = gray[first_crop_line_height + 10:second_crop_line_height - 10, :]
    return segmented_image, segmented_image_original


def detect_sentences(segmented_image, segmented_image_original):
    """
    This function detect sentences and return a list of detected sentences.
    """
    # Apply erosion to remove noise.
    kernel = np.ones((3, 5), np.uint8)
    img_erosion = cv2.erode(segmented_image, kernel, iterations=3)

    # Apply dilation to connect sentence together as one block
    # Taking a matrix of size 3,10 as the kernel 
    kernel = np.ones((2, 25), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=11)

    # Find contours
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Find bounded rectangles surround the contours
    bounding_rectangles = [None] * len(contours)
    for i in range(len(contours)):
        bounding_rectangles[i] = cv2.boundingRect(contours[i])

    sentences = []
    # Loop on the contours and crop each sentence
    i = len(contours) - 1
    while i >= 0:
        difference_in_width = bounding_rectangles[i][2]
        difference_in_height = bounding_rectangles[i][3]
        if difference_in_width > 600 and difference_in_height > 50:
            initial_height = int(bounding_rectangles[i][1]) - 20
            if initial_height < 0:
                initial_height = 0
            seg_sentence = np.copy(segmented_image_original[initial_height:int(bounding_rectangles[i][1])
                                                                           + difference_in_height,
                                   int(bounding_rectangles[i][0]) + 50:int(bounding_rectangles[i][0])
                                                                       + difference_in_width - 50])
            sentences.append(seg_sentence)
        i -= 1
    return sentences


def preprocessing(img):
    """
    This function pre-process the image and return list of sentences.
    """
    # Extract hand written part
    segmented_image, segmented_image_original = extract_hand_written(img)
    # Apply noise removal.
    segmented_image_original = cv2.blur(segmented_image_original, (3, 3))
    # Extract sentences
    sentences = detect_sentences(segmented_image, segmented_image_original)
    return sentences
