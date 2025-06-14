import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision



image = torchvision.io.read_image("Path-of-The-Image")
image = image.numpy()

upscaled = cv2.resize(image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

# Initial Denoising
denoised_initial = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(denoised_initial, kernel, iterations=1)

# Erosion
eroded = cv2.erode(dilated, kernel, iterations=1)


#####----------------------------------EDGE DETECTION--------------------------------------------########3


#----------------------------------------------------------------------------------------------------
# Edge Detection: Sobel + Gaussian Blur
# Convert to grayscale
gray = cv2.cvtColor(eroded, cv2.COLOR_RGB2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Sobel Edge Detection (both x and y gradients)
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude of the gradient
edges_sobel = cv2.magnitude(sobelx, sobely)
edges_sobel = cv2.convertScaleAbs(edges_sobel)

# Final Denoising on the Sobel output
denoised_final = cv2.fastNlMeansDenoising(edges_sobel, None, 10, 7, 21)


