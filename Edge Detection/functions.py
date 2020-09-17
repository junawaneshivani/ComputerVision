import numpy as np


def correlation(image, kernel):
    '''
    Compares similarity of two sets of data
    '''
    return filter2d(image, kernel)    
    

def convolution(image, kernel):
    '''
    Expresses amount of overlap of one function as it is shifted over another function
    '''
    # rotate kernel by 180 degree
    kernel = np.rot90(kernel, k=2)
    print(kernel)
    return filter2d(image, kernel)


def filter2d(image, kernel):
    '''
    Parameters
    ----------
    image : grey scale image
    kernel : 2d filter
    
    Returns
    -------
    filtered_image : ndarray
    sum: int
    '''

    # exceptions
    if len(kernel.shape)!=2:
        raise Exception("Only 2d filters allowed")
    if len(image.shape)!=2:
        raise Exception("Only 2d images(gray scale) allowed")

    # shapes
    kernel_x, kernel_y = kernel.shape
    image_x,  image_y  = image.shape

    if kernel_x%2==0 or kernel_y%2==0:
        raise Exception("kernel height and width cannot be even")

    # calculate required padding
    padding_x = (kernel_x - 1) // 2
    padding_y = (kernel_y - 1) // 2

    # pad the image for same output size
    padded_image = np.pad(array=image, pad_width=((padding_x, padding_x), (padding_y, padding_y)), mode='edge')

    # output array
    filtered_image = np.zeros_like(image)

    # iterate over image
    X, Y = padded_image.shape
    sum = 0
    for x in range(padding_x, X-2*padding_x):
        for y in range(padding_y, Y-2*padding_y):          
            # iterate over kernel
            filtered_image[x-1, y-1] = np.sum(
                np.multiply(
                    padded_image[x-padding_x:x+padding_x+1, y-padding_y:y+padding_y+1], 
                    kernel
                )
            )
    return filtered_image


def zero_crossings(image, threshold):
    x, y = image.shape
    output_image = np.zeros_like(image)
    for i in range(1, x-1):
        for j in range(1, y-1):
            # left right
            if opp_sign(image[i][j-1], image[i][j+1]):
                if abs(image[i][j-1] - image[i][j+1]) > threshold:
                    #print(abs(image[i][j-1]-image[i][j+1]))
                    #image[i][j] = 0
                    continue

            # up down
            if opp_sign(image[i-1][j], image[i+1][j]):
                if abs(image[i-1][j] - image[i+1][j]) > threshold:
                    #image[i][j] = 0
                    continue
            # diagonal
            if opp_sign(image[i+1][j-1], image[i-1][j+1]):
                if abs(image[i+1][j-1] - image[i-1][j+1]) > threshold:
                    #image[i][j] = 0
                    continue
            if opp_sign(image[i-1][j-1], image[i+1][j+1]):
                if abs(image[i-1][j-1] - image[i+1][j+1]) > threshold:
                    #image[i][j] = 0
                    continue

            output_image[i][j] = 255

    return output_image


def opp_sign(x, y):
    True if (x<0 and y>0) or (x>0 and y<0) else False


def _test_case():
    from skimage import io
    import cv2
    image = io.imread("https://junawaneshivani.github.io/images/profile.jpg", as_gray=True)
    image = cv2.resize(image, (400, 400))
    cv2.imshow("Original", image)
    cv2.waitKey(0)
    corr_image = correlation(image, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    cv2.imshow("Corr Image", corr_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    _test_case()
