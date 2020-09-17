import functions
import numpy as np
import math


def prewitt_operator(image):
    # mask is combination of smoothing and derivative
    v_mask = np.array([ [1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    v_edges = functions.convolution(image, kernel=v_mask)
    h_mask = np.array([ [1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])
    h_edges = functions.convolution(image, kernel=h_mask)
    output = np.sqrt(v_edges**2 + h_edges**2)

    threshold = np.absolute(output).mean() * 1.5
    output[output <= threshold] = 255
    return output


def sobel_operator(image):
    # mask is combination of smoothing and derivative
    v_mask = np.array([ [1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    v_edges = functions.convolution(image, kernel=v_mask)
    h_mask = np.array([ [1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    h_edges = functions.convolution(image, kernel=h_mask)
    output = np.sqrt(v_edges**2 + h_edges**2)

    threshold = np.absolute(output).mean() * 1.5
    output[output <= threshold] = 255
    return output


def marr_hildreth(image):
    # LOG masks can be represented in multiple ways, this is most common one
    # https://www.youtube.com/watch?v=L-aVtLns9wY
    mask = np.array([   [ 0, 0,   1, 0, 0],
                        [ 0, 1,   2, 1, 0],
                        [ 1, 2, -16, 2, 1],
                        [ 0, 1,   2, 1, 0],
                        [ 0, 0,   1, 0, 0] ])   
    output = functions.convolution(image, kernel=mask)
    max_value = np.max(output)
    threshold = 0.15 * max_value
    # TODO: Find zero crossings
    # TODO: Threshold the zero-crossings to keep only those strong ones 
    output = functions.zero_crossings(output, threshold)
    return output


def canny(image):
    pass


def _test_case():
    from skimage import io
    import cv2
    import matplotlib.pyplot as plt
    image = io.imread("https://junawaneshivani.github.io/images/profile.jpg", as_gray=True)
    image = cv2.resize(image, (400, 400))
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap="gray")
    output_image = prewitt_operator(image)
    plt.subplot(2, 3, 2).set_title("prewitt")
    plt.imshow(output_image, cmap="gray")
    #print(output_image)
    output_image = sobel_operator(image)
    plt.subplot(2, 3, 3).set_title("sobel")
    plt.imshow(output_image, cmap="gray")
    output_image = marr_hildreth(image)
    plt.subplot(2, 3, 5).set_title("marr hildreth")
    plt.imshow(output_image, cmap=plt.cm.gray)
    plt.show()

if __name__ == "__main__":
    _test_case()