import os
from multiprocessing import Pool
from skimage import feature
import numpy as np
import time


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        start = time.time()
        # to build the histogram of patterns
#         lbp = feature.local_binary_pattern(image, self.numPoints,
#             self.radius, method="default")
        lbp = lbp_custom(image)
        print("lbp time", time.time() - start)
        start = time.time()

        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        print("histogram time", time.time() - start)
        start = time.time()
        # normalize the histogram
        
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        print("normalization time", time.time() - start)
        start = time.time()
        return hist


img_lbp = 0
def lbp(i, image, width):
    global img_lbp
    for j in range(0, width): 
        img_lbp[i, j] = lbp_calculated_pixel(image, i, j) 

def lbp_custom(image):
    global img_lbp
    height, width = image.shape 
    img_lbp = np.zeros((height, width), 
                   np.uint8)
    pool = Pool(os.cpu_count())
    pool.map(lbp, range(0, height), image, width)
    pool.join()
    pool.close()
    return img_lbp

def get_pixel(img, center, x, y): 
      
    new_value = 0
      
    try: 
        # If local neighbourhood pixel  
        # value is greater than or equal 
        # to center pixel values then  
        # set it to 1 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
        # Exception is required when  
        # neighbourhood value of a center 
        # pixel value is null i.e. values 
        # present at boundaries. 
        pass
      
    return new_value 
   
# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y, r = 3): 
   
    center = img[x][y] 
   
    val_ar = [] 
      
    # top_left 
    val_ar.append(get_pixel(img, center, x-r, y-r)) 
      
    # top 
    val_ar.append(get_pixel(img, center, x-r, y)) 
      
    # top_right 
    val_ar.append(get_pixel(img, center, x-r, y + r)) 
      
    # right 
    val_ar.append(get_pixel(img, center, x, y + r)) 
      
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + r, y + r)) 
      
    # bottom 
    val_ar.append(get_pixel(img, center, x + r, y)) 
      
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + r, y-r)) 
      
    # left 
    val_ar.append(get_pixel(img, center, x, y-r)) 
       
    # Now, we need to convert binary 
    # values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val 