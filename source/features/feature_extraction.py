import time
import os
import cv2
from matplotlib import pyplot as plt
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# TODO:  this will be removed later from here
# Load an color image in grayscale
imgs = []
directory = './temp2/'
test_count = len([name for name in os.listdir(directory) if os.path.isfile(directory+name)])

for i in range(1,test_count + 1):
    imgs.append(cv2.imread('./temp2/test'+str(i)+'.JPG'))
    imgs[i-1] = cv2.cvtColor(imgs[i-1], cv2.COLOR_BGR2GRAY)

# Load an color image in grayscale
imgs_1 = []
directory = './temp2/1/'
count_1 = len([name for name in os.listdir(directory) if os.path.isfile(directory+name)])
for i in range(1,count_1+1):
    imgs_1.append(cv2.imread('./temp2/1/'+str(i)+'.PNG'))
    imgs_1[i-1] = cv2.cvtColor(imgs_1[i-1], cv2.COLOR_BGR2GRAY)
directory = './temp2/2/'
count_2 = len([name for name in os.listdir(directory) if os.path.isfile(directory + name)])
imgs_2 = []
for i in range(1,count_2+1):
    imgs_2.append(cv2.imread('./temp2/2/'+str(i)+'.PNG'))
    imgs_2[i-1] = cv2.cvtColor(imgs_2[i-1], cv2.COLOR_BGR2GRAY)
imgs_3 = []
directory = './temp2/3/'
count_3 = len([name for name in os.listdir(directory) if os.path.isfile(directory+name)])

for i in range(1,count_3+1):
    imgs_3.append(cv2.imread('./temp2/3/'+str(i)+'.PNG'))
    imgs_3[i-1] = cv2.cvtColor(imgs_3[i-1], cv2.COLOR_BGR2GRAY)

# TODO:  this will be removed later to here

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
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