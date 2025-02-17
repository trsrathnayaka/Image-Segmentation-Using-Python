import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_images(images, titles, cmap='gray'):
    """Display images in a grid layout"""
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Make Gaussian Blur
def apply_gaussian_blur(img, kernel_size=5, sigma=1):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


# Make Sobel Gradient
def sobel_filters(img):
    """Compute Sobel gradients and gradient direction"""
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255  
    # Normalize

    theta = np.arctan2(Iy, Ix)
    return G, theta


# Make Non-Maximum Suppression
def non_max_suppression(img, theta):
    """Suppress non-maximum pixels based on gradient direction"""
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.uint8)
    angle = np.rad2deg(theta) % 180  

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # Define gradient direction cases
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass
    return Z


# Make Double Threshold
def threshold(img, low_ratio=0.05, high_ratio=0.15):
    """Apply double threshold to identify strong and weak edges"""
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)

    weak = 50
    strong = 255

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img >= low_threshold) & (img < high_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong


# Make Hysteresis Edge Tracking
def hysteresis(img, weak=50, strong=255):
    """Ensure weak edges connected to strong edges are preserved"""
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if (img[i + 1, j - 1] == strong or img[i + 1, j] == strong or img[i + 1, j + 1] == strong or
                        img[i, j - 1] == strong or img[i, j + 1] == strong or
                        img[i - 1, j - 1] == strong or img[i - 1, j] == strong or img[i - 1, j + 1] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


# Main Function Execution
if __name__ == '__main__':
    # Read image (Convert to grayscale)
    img = cv2.imread('assets/charlie.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found! Check the file path.")

    # Processing pipeline
    blurred = apply_gaussian_blur(img)
    gradient, theta = sobel_filters(blurred)
    suppressed = non_max_suppression(gradient, theta)
    thresholded, weak, strong = threshold(suppressed)
    final_edges = hysteresis(thresholded.copy(), weak, strong)

    # Display results
    images = [img, blurred, gradient, suppressed, thresholded, final_edges]
    titles = [
        'Original Image',
        'Gaussian Blurred',
        'Sobel Gradient',
        'Non-Max Suppression',
        'Double Threshold',
        'Final Edges (Hysteresis)'
    ]

    plot_images(images, titles)

#Name: TRS Rathnayaka
#SID: 23912
