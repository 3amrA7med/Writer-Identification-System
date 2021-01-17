import numpy as np
from skimage import feature

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
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        
        lbp = self.get_lbp(image)
        #lbp = feature.local_binary_pattern(image, self.numPoints,
            #self.radius, method="default")
        # lbp = lbp_custom(image)
        
        
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))

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


    def shift(self, img, shift) -> np.ndarray:
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