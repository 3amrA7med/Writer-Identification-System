import numpy as np
from commonfunctions import show_images
from skimage import feature

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
def lbp_calculated_pixel(img, x, y): 

    center = img[x][y] 

    val_ar = [] 
        
    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
        
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
        
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
        
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
        
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
        
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
        
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
        
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
        
    # Now, we need to convert binary 
    # values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 

    val = 0
        
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
            
    return val 

def custom_lbp(img):
    height, width = img.shape 
    img_lbp = np.zeros((height, width), 
                   np.uint8) 
   
    for i in range(0, height): 
        for j in range(0, width): 
            img_lbp[i, j] = lbp_calculated_pixel(img, i, j) 
    return img_lbp



class LocalBinaryPatterns:
    def __init__(self, accuracy):
        # store the number of points and radius
        self.accuracy = accuracy
        if self.accuracy == 1:
            self.numPoints = 8
        elif self.accuracy == 2:
            self.numPoints = 16
        else:
            self.numPoints = 24

    def describe(self, image, eps=1e-7):
        """
        compute the Local Binary Pattern representation
        of the image, and then use the LBP representation
        to build the histogram of patterns
        :param image: image used to extract features
        :param eps:
        :return: histogram
        """

        lbp = self.get_lbp(image)
        # lbp = custom_lbp(image)
        # lbp = feature.local_binary_pattern(image, 8, 3, method='default')

        # Find histrogram
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

    def get_lbp(self, img):
        # Get image dimensions
        height, width = img.shape

        # Initialize LBP image
        lbp = np.zeros((height, width), dtype=np.uint8)

        # Directions
        v = 3
        j = 2
        k = 1
        if self.accuracy == 1:
            dx = [0, v, v, v, 0, -v, -v, -v]
            dy = [v, v, 0, -v, -v, -v, 0, v]
        elif self.accuracy == 2:
            dx = [0, v, v, v, 0, -v, -v, -v, 0, j, j, j, 0, -j, -j, -j]
            dy = [v, v, 0, -v, -v, -v, 0, v, j, j, 0, -j, -j, -j, 0, j]
        else:
            dx = [0, v, v, v, 0, -v, -v, -v, 0, j, j, j, 0, -j, -j, -j, 0, k, k, k, 0, -k, -k, -k]
            dy = [v, v, 0, -v, -v, -v, 0, v, j, j, 0, -j, -j, -j, 0, j, k, k, 0, -k, -k, -k, 0, k]

        # Loop over the 8 neighbors
        for i in range(self.numPoints):
            view_shf = self.shift(img, (dy[i], dx[i]))
            view_img = self.shift(img, (-dy[i], -dx[i]))
            view_lbp = self.shift(lbp, (-dy[i], -dx[i]))
            res = (view_img >= view_shf)
            view_lbp |= (res.view(np.uint8) << i)
        return lbp

    def shift(self, img, shift):
        r, c = shift[0], shift[1]
        if r >= 0:
            ret = img[r:, :]
        else:
            ret = img[0:r, :]

        if c >= 0:
            ret = ret[:, c:]
        else:
            ret = ret[:, 0:c]
        return ret

