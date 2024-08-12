import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Lab 2 Task 1.1

# # Load the grayscale image
# image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('input_image_GRAYSCALE.jpg', image)
#
# # Apply Otsu's method for binarization
# _, binary_image_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# # Display the original and binarized images
# cv2.imshow('Original Image', image)
# cv2.imshow('Otsu Binarized Image', binary_image_otsu)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Save the binarized image
# cv2.imwrite('otsu_binarized_image.jpg', binary_image_otsu)


#
# # Lab 2 Task 1.2
# def calculate_histogram(image):
#     histogram, _ = np.histogram(image, bins=np.arange(257))
#     return histogram
#
# def calculate_probabilities(histogram, total_pixels):
#     probabilities = histogram / total_pixels
#     cumulative_probabilities = np.cumsum(probabilities)
#     cumulative_means = np.cumsum(probabilities * np.arange(256))
#     global_mean = cumulative_means[-1]
#     return probabilities, cumulative_probabilities, cumulative_means, global_mean
#
# def calculate_intra_class_variance(cumulative_probabilities, cumulative_means, global_mean):
#     intra_class_variance = np.zeros(256)
#     for t in range(256):
#         if cumulative_probabilities[t] == 0 or cumulative_probabilities[t] == 1:
#             intra_class_variance[t] = np.inf
#         else:
#             class1_prob = cumulative_probabilities[t]
#             class2_prob = 1 - class1_prob
#             class1_mean = cumulative_means[t] / class1_prob
#             class2_mean = (global_mean - cumulative_means[t]) / class2_prob
#             intra_class_variance[t] = class1_prob * class2_prob * (class1_mean - class2_mean) ** 2
#     optimal_threshold = np.argmax(intra_class_variance)
#     return intra_class_variance, optimal_threshold
#
# def apply_threshold(image, optimal_threshold):
#     binary_image = (image > optimal_threshold).astype(np.uint8) * 255
#     return binary_image
#
# # Load the grayscale image
# image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Calculate histogram
# histogram = calculate_histogram(image)
#
# # Calculate probabilities and cumulative sums
# total_pixels = image.size
# probabilities, cumulative_probabilities, cumulative_means, global_mean = calculate_probabilities(histogram, total_pixels)
#
# # Calculate intra-class variance and find the optimal threshold
# intra_class_variance, optimal_threshold = calculate_intra_class_variance(cumulative_probabilities, cumulative_means, global_mean)
#
# # Plot the intra-class variance
# plt.plot(intra_class_variance)
# plt.title('Intra-class Variance vs. Threshold')
# plt.xlabel('Threshold')
# plt.ylabel('Intra-class Variance')
# plt.show()
#
# # Apply the threshold to binarize the image
# otsu_binarized_image_custom = apply_threshold(image, optimal_threshold)
# cv2.imshow('Otsu Binarized Image Custom', otsu_binarized_image_custom)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Save the binarized image
# cv2.imwrite('otsu_binarized_image_custom.jpg', otsu_binarized_image_custom)
#
# print("Global Mean:", global_mean)
# print("Optimal Threshold:", optimal_threshold)


# # Lab 2 Task 2.1

def my_canny(image, low_threshold, high_threshold, kernel_size=3, sigma=1):
    # Step 1: Gaussian blur
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
        return kernel / np.sum(kernel)

    def gaussian_blur(image, kernel_size, sigma):
        kernel = gaussian_kernel(kernel_size, sigma)
        return convolve(image.astype(np.float64), kernel)

    smoothed_image = gaussian_blur(image, kernel_size, sigma)

    # Step 2: Compute gradients (Sobel operators)
    def sobel_filters(image):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        grad_x = convolve(image, sobel_x)
        grad_y = convolve(image, sobel_y)

        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)

        return grad_magnitude, grad_direction

    grad_magnitude, grad_direction = sobel_filters(smoothed_image)

    # Step 3: Non-maximum suppression
    def non_max_suppression(grad_magnitude, grad_direction):
        suppressed = np.zeros(grad_magnitude.shape, dtype=np.float64)
        for i in range(1, grad_magnitude.shape[0] - 1):
            for j in range(1, grad_magnitude.shape[1] - 1):
                direction = grad_direction[i, j]
                mag = grad_magnitude[i, j]

                # Determine neighboring pixels in the gradient direction
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    before, after = grad_magnitude[i, j - 1], grad_magnitude[i, j + 1]
                elif 22.5 <= direction < 67.5:
                    before, after = grad_magnitude[i - 1, j - 1], grad_magnitude[i + 1, j + 1]
                elif 67.5 <= direction < 112.5:
                    before, after = grad_magnitude[i - 1, j], grad_magnitude[i + 1, j]
                else:
                    before, after = grad_magnitude[i - 1, j + 1], grad_magnitude[i + 1, j - 1]

                # Compare neighboring pixels and keep only local maxima
                if mag >= before and mag >= after:
                    suppressed[i, j] = mag

        return suppressed

    suppressed_image = non_max_suppression(grad_magnitude, grad_direction)

    # Step 4: Double thresholding and edge tracking by hysteresis
    def apply_threshold(image, low_threshold, high_threshold):
        strong_edges = (image >= high_threshold)
        weak_edges = (image < high_threshold) & (image >= low_threshold)
        return strong_edges.astype(np.uint8) * 255, weak_edges.astype(np.uint8) * 255

    low_threshold = int(low_threshold)
    high_threshold = int(high_threshold)

    strong_edges, weak_edges = apply_threshold(suppressed_image, low_threshold, high_threshold)

    # Step 5: Edge tracking by hysteresis
    def hysteresis_thresholding(strong_edges, weak_edges):
        h, w = strong_edges.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak_edges[i, j]:
                    if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                        strong_edges[i, j] = 255
                    else:
                        weak_edges[i, j] = 0

        return strong_edges

    edges_image = hysteresis_thresholding(strong_edges, weak_edges)

    return edges_image.astype(np.uint8)


# Load an image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Parameters for Canny edge detection
low_threshold = 50
high_threshold = 100

# Apply your custom Canny method
edges_custom = my_canny(image, low_threshold, high_threshold)

# Apply OpenCV's Canny method for comparison
edges_opencv = cv2.Canny(image, low_threshold, high_threshold)

# Save custom Canny result
cv2.imwrite('custom_canny_output.jpg', edges_custom)

# Save OpenCV Canny result
cv2.imwrite('opencv_canny_output.jpg', edges_opencv)

# Display results using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges_custom, cmap='gray')
plt.title('Custom Canny')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges_opencv, cmap='gray')
plt.title('OpenCV Canny')
plt.axis('off')

plt.tight_layout()
plt.show()
