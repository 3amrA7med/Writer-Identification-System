import os
import time
from pathlib import Path
import numpy as np
import cv2

def horizontalProjection(img):
    h, w = img.shape
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w]
        sumRows.append(int(np.sum(row)/255))
    return sumRows


def extract_hand_written(img):
    """
    This function returns original cropped image and dilated cropped image 
    for the handwritten part only and exclude any unused parts.
    """
    # Extract grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Apply binary thresholding using otsu's method to inverse the background and content color.
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    # DEBUG: show_images([thresh_inv], ["Thresholding"]) 
    
    # Apply noise removal.
    #blur = cv2.GaussianBlur(thresh_inv,(7,7),0)
    blur = cv2.blur(thresh_inv,(7,7))
    #DEBUG: show_images([blur],["blur image"])
    
    # Apply thresholding for better and more general output.
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #DEBUG: show_images([binary],["Binary image"])
    
    # Apply dilation to the image to make sure that horizontal lines are continuous.
    # Taking a matrix of size 4,2 as the kernel 
    kernel = np.ones((4,1), np.uint8) 
    img_dilation = cv2.dilate(binary, kernel, iterations=3)
    #DEBUG: show_images([img_dilation],["Dilation"])
    
    # Find lines by using contour
    # Find contours
    contours, hierarchy = cv2.findContours(img_dilation,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    # Find bounded rectangles surround the contours
    boundRect = [None]*len(contours)
    for i in range(len(contours)):
        boundRect[i] = cv2.boundingRect(contours[i])

    lines_pos = []
    height_diff = []
    i = 0
    while i < len(contours):
        # print(boundRect[i][2], boundRect[i][1])
        if boundRect[i][2] > 700:
            #cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]),
              #int(boundRect[i][1]+boundRect[i][3])), (0,0,0), 2)
            #print(boundRect[i][1])
            lines_pos.append(boundRect[i][1])
            height_diff.append(boundRect[i][3])
            i += 10
        i += 1
    
    third = first = second = -1 
    max_index = mid_index = 0 
    for i in range(len(lines_pos)):
        if (lines_pos[i] > first):       
            third = second 
            second = first 
            first = lines_pos[i]
            max_index = i 
        elif (lines_pos[i] > second): 
            third = second 
            second = lines_pos[i] 
            mid_index = i
        elif (lines_pos[i] > third): 
            third = lines_pos[i] 
            
    # Take the line before hand written text directly.
    first_crop_line_height = lines_pos[mid_index] + height_diff[mid_index]
    # Take the line after hand written text directly.
    second_crop_line_height = np.max(lines_pos) 
    
    # print(first_crop_line_height+10, second_crop_line_height-10)
    # Crop the original image and the dilated image
    segIm = np.copy(img_dilation[first_crop_line_height+10:second_crop_line_height-10,:])
    segIm_original = np.copy(img[first_crop_line_height+10:second_crop_line_height-10,:])
    #DEBUG:show_images([segIm, segIm_original],["Segmented", "Segmented 2"])
    
    return segIm, segIm_original

def detect_sentences(segIm, segIm_original):
    """
    This function detect sentences and return a list of detected sentences.
    """
    # show_images([segIm],["Before"])
    # Apply erosion to remove noise.
    kernel = np.ones((3,5), np.uint8) 
    img_erosion = cv2.erode(segIm, kernel, iterations=3) 
    # DEBUG:show_images([img_erosion], ["After erosion"]) 

    # Apply dilation to connect sentence togther as one block
    # Taking a matrix of size 3,10 as the kernel 
    kernel = np.ones((2, 20), np.uint8) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=11)
    # DEBUG:
    # show_images([img_dilation], ["After dilation"])

    # Find contours
    contours, hierarchy = cv2.findContours(img_dilation,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    # Find bounded rectangles surround the contours
    boundRect = [None]*len(contours)
    for i in range(len(contours)):
        boundRect[i] = cv2.boundingRect(contours[i])
        
    number_of_countours = 0
    sentences = []

    # Loop on the contours and crop each sentence
    i = len(contours) - 1
    while i >= 0:
        difference_in_width = boundRect[i][2]
        difference_in_height = boundRect[i][3]
        if(difference_in_width > 250 and difference_in_height > 50):
            #print(boundRect[i])
            number_of_countours += 1
            #cv2.rectangle(segIm_original, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]),
             #                                           int(boundRect[i][1]+boundRect[i][3])), (0,0,0), 2)
            initial_height = int(boundRect[i][1])-20
            if initial_height < 0:
                initial_height = 0
            seg_sentence = np.copy(segIm_original[initial_height:int(boundRect[i][1]) + difference_in_height,
                                                  int(boundRect[i][0]):int(boundRect[i][0]) + difference_in_width])
            sentences.append(seg_sentence)
            #DEBUG: show_images([cv2.cvtColor(seg_sentence, cv2.COLOR_BGR2RGB)], ["sentence"+str(number_of_countours)])
        i -= 1


    #show_images([img],["Image"])
    #show_images([img_dilation], ["After dilation"])
    #show_images([gray],["Grayscale"])
    #show_images([binary], ["Otsu thresholding"])
 
    #DEBUG:cv2.drawContours(segIm_original, contours, -1, (0, 0, 0), 3) 
    #DEBUG:show_images([segIm_original],["With contours"])
    return sentences

def preprocessing(img):
    """
    This function preprocess the image and return list of sentences.
    """
    
    # Extract hand written part
    segIm, segIm_original = extract_hand_written(img)
    #DEBUG: show_images([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)],["Original Image"])
    # Apply noise removal.
    blur = cv2.blur(segIm_original,(7,7))
    # Extract sentences
    sentences = detect_sentences(segIm, segIm_original)
    for i in range(len(sentences)):
        sentences[i] = cv2.cvtColor(sentences[i], cv2.COLOR_BGR2GRAY)

    return sentences
    