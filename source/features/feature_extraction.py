import time
import os
import cv2
from matplotlib import pyplot as plt
from skimage import feature
from source.features.LBP import LocalBinaryPatterns


def feature_extractor(imgs_1, imgs_2, imgs_3):
    # initialize the local binary patterns descriptor along with
    # the data and label lists
    start = time.time()
    desc = LocalBinaryPatterns(8, 3)
    data = []
    labels = []

    # loop over the training images
    for img in imgs_1:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append("1")
        data.append(hist)
    for img in imgs_2:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append("2")
        data.append(hist)
    for img in imgs_3:
        # load the image, convert it to grayscale, and describe it
        hist = desc.describe(img)
        # extract the label from the image path, then update the
        # label and data lists
        labels.append("3")
        data.append(hist)
    end = time.time()
    print("Writers feature extraction time:" + str(end - start))
    return data, labels, desc


def test(model, imgs, desc):
    # loop over the testing images
    start = time.time()
    results = []
    for image in imgs:
        hist = desc.describe(image)
        prediction = model.predict(hist.reshape(1, -1))
        # # display the image and the prediction
        # cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
        # plt.imshow(image,cmap='Greys_r')
        # plt.title('test')
        # plt.show()
        results.append(prediction[0])
    print("Testing time:" + str(time.time() - start))
    return results
