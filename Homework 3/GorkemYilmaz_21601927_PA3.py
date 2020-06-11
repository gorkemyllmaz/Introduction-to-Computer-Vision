"""
Author: Görkem Yılmaz
Date: April 15, 2020
Introduction to Computer Vision 
Homework 3
"""
# imports
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy import misc

"""
PART 1.1
"""
# Convert edge1.png, edge2.png and edge3.png into grayscale images
edge1 = rgb2gray(mpimg.imread("edge1.png"))
edge2 = rgb2gray(mpimg.imread("edge2.png"))
edge3 = rgb2gray(mpimg.imread("edge3.png"))
# Remove Noise with Gaussian Blur
img1 = cv2.GaussianBlur(edge1,(3,3),0)
img2 = cv2.GaussianBlur(edge2,(3,3),0)
img3 = cv2.GaussianBlur(edge3,(3,3),0)

# SOBEL EDGE DETECTION
def sobelEdgeDetection(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return sobelx, sobely

# PLOT IMAGES
# img1
sobelx, sobely = sobelEdgeDetection(img1)
cv2.imshow("Original Image", img1)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.waitKey(0)

# img2
sobelx, sobely = sobelEdgeDetection(img2)
cv2.imshow("Original Image", img2)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.waitKey(0)

# img3
sobelx, sobely = sobelEdgeDetection(img3)
cv2.imshow("Original Image", img3)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.waitKey(0)


# PREWITT EDGE DETECTION
def prewittEdgeDetection(img):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    prewittx = cv2.filter2D(img, -1, kernelx)
    prewitty = cv2.filter2D(img, -1, kernely)

    return prewittx, prewitty
    
# PLOT IMAGES

# img1
prewittx, prewitty = prewittEdgeDetection(img1)
cv2.imshow("Original Image", img1)
cv2.imshow("Prewitt X", prewittx)
cv2.imshow("Prewitt Y", prewitty)
cv2.waitKey(0)

# img2
prewittx, prewitty = prewittEdgeDetection(img2)
cv2.imshow("Original Image", img2)
cv2.imshow("Prewitt X", prewittx)
cv2.imshow("Prewitt Y", prewitty)
cv2.waitKey(0)

# img3
prewittx, prewitty = prewittEdgeDetection(img3)
cv2.imshow("Original Image", img3)
cv2.imshow("Prewitt X", prewittx)
cv2.imshow("Prewitt Y", prewitty)
cv2.waitKey(0)

"""
PART 1.2
"""
# CANNY EDGE DETECTION AND PLOTTING
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    return Z

def threshold(img,lowThreshold, highThreshold, weak_pixel, strong_pixel):

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res)

def hysteresis(img, weak_pixel, strong_pixel):

    M, N = img.shape
    weak = weak_pixel
    strong = strong_pixel

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img


def cannyEdgeDetector(img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):

    img_smoothed = convolve(img, gaussian_kernel(kernel_size, sigma))
    gradientMat, thetaMat = sobel_filters(img_smoothed)
    nonMaxImg = non_max_suppression(gradientMat, thetaMat)
    thresholdImg = threshold(nonMaxImg,lowthreshold, highthreshold, weak_pixel, strong_pixel)
    img_final = hysteresis(thresholdImg, weak_pixel, strong_pixel)

    canny_img = np.array(img_final, dtype=np.uint8)

    return canny_img
    
# PLOT

# img1
canny_img = cannyEdgeDetector(img1, 1, 5, 75, 255, 0.05, 40)
cv2.imshow("Original Image", img1)
cv2.imshow("Canny Edge Detected Image", canny_img)
cv2.waitKey(0)

# img2
canny_img = cannyEdgeDetector(img2, 0.2, 5, 75, 255, 0.05, 20)
cv2.imshow("Original Image", img2)
cv2.imshow("Canny Edge Detected Image", canny_img)
cv2.waitKey(0)

# img3
canny_img = cannyEdgeDetector(img3, 2, 5, 75, 255, 0.05, 30)
cv2.imshow("Original Image", img3)
cv2.imshow("Canny Edge Detected Image", canny_img)
cv2.waitKey(0)

"""
PART 2
"""
img = cv2.imread("edge1.png")
cannied = cv2.Canny(img, 50, 200, None, 3)
grayCannied = cv2.cvtColor(cannied, cv2.COLOR_GRAY2BGR)
# lines = cv2.HoughLines(cannied, 1, np.pi / 180, 72) # hough.png
lines = cv2.HoughLines(cannied, 1, np.pi / 180, 316) # edge1.png, edge2.png
# lines = cv2.HoughLines(cannied, 1, np.pi / 180, 144) # hough3.png

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        point1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        point2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(grayCannied, point1, point2, (0,0,255), 3, cv2.LINE_AA)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", grayCannied)
cv2.waitKey(0)