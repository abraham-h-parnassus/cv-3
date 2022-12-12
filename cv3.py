import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage.color import rgb2gray
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from skimage.util import random_noise

from utils import canvas_to_np


# Get an image
# Make sure it's half toned
# Increase blur / noise from 0 to S
#    Detect Edges with Canny
#    Detect linesn with Hough
#    Display
# Compare
#

def process(image, detect_circles=False):
    # Show reference image
    # plt.imshow(image)
    # pyplot.show()
    noises = [0.00, 0.03, 0.06]
    result = []
    for variance in noises:
        noisy_image = random_noise(image, var=variance)
        if len(noisy_image.shape) == 2:
            grayed_image = noisy_image
        else:
            grayed_image = rgb2gray(noisy_image)
        cannied = canny(grayed_image, 0)

        # Generating figure 1
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax = axes.ravel()
        ax[0].imshow(noisy_image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        if not detect_circles:
            ax[1].imshow(image, cmap=cm.gray)
            ax[1].set_ylim((image.shape[0], 0))
            ax[1].set_axis_off()
            ax[1].set_title('Detected lines')
            h, theta, d = hough_line(cannied)
            peaks = zip(*hough_line_peaks(h, theta, d))
            for _, angle, dist in peaks:
                (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
                ax[1].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

        else:
            ax[1].set_ylim((image.shape[0], 0))
            ax[1].set_axis_off()
            ax[1].set_title('Detected shapes')
            ax[1].imshow(np.full((image.shape[0], image.shape[1], 3), (0, 0, 0)))
            try_radii = np.arange(20, 40)
            h = hough_circle(cannied, try_radii)
            accum, cx, cy, rad = hough_circle_peaks(h, try_radii)
            for center_y, center_x, radius in zip(cy, cx, try_radii):
                circle = plt.Circle((center_x, center_y), int(radius), color='red', fill=False)
                ax[1].add_patch(circle)
        # ax[1].imshow(reference_image)
        plt.tight_layout()
        data = canvas_to_np(fig)
        result.append(data)
    return result



if __name__ == '__main__':
    image = cv2.imread("/images/lena.png")
    process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
