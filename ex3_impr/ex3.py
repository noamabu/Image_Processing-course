import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplacian_pyramid(image, num_levels):
    pyramid = []
    current_level = image.copy()
    for _ in range(num_levels):
        lower_res = cv2.pyrDown(current_level)
        expanded = cv2.pyrUp(lower_res, dstsize=(current_level.shape[1], current_level.shape[0]))
        diff = cv2.subtract(current_level, expanded)
        pyramid.append(diff)
        current_level = lower_res
    pyramid.append(current_level)
    return pyramid


def gaussian_pyramid(image, num_levels):
    pyramid = [image]
    current_level = image.copy()
    for _ in range(num_levels):
        current_level = cv2.pyrDown(current_level)
        pyramid.append(current_level)
    return pyramid


def blend_images(laplacian_pyr1, laplacian_pyr2, gaussian_pyr_mask):
    blended_pyr = []
    for i in range(len(gaussian_pyr_mask)):
        blended_img = laplacian_pyr1[i] * gaussian_pyr_mask[i] + laplacian_pyr2[i] * (1 - gaussian_pyr_mask[i])
        blended_pyr.append(blended_img)
    return blended_pyr


def sum_pyramid(pyramid):
    reconstructed_img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        expanded_image = cv2.pyrUp(reconstructed_img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        reconstructed_img = pyramid[i] + expanded_image
    return reconstructed_img


def images_blending(path_image_1, path_image_2, path_mask):
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)
    mask = cv2.imread(path_mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)/255
    color_image = []
    for i in range(3):
        image_1 = image1[:, :, i]
        image_1 = image_1/255
        image_2 = image2[:, :, i]
        image_2 = image_2/255
        laplacian_pyramid_image1 = laplacian_pyramid(image_1, 5)
        laplacian_pyramid_image2 = laplacian_pyramid(image_2, 5)
        gaussian_pyramid_mask = gaussian_pyramid(mask, 5)
        blended_pyramid = blend_images(laplacian_pyramid_image1, laplacian_pyramid_image2, gaussian_pyramid_mask)
        color_image.append(sum_pyramid(blended_pyramid))
    blended_image = np.stack([color_image[2], color_image[1], color_image[0]], axis=-1)
    plt.imshow(blended_image)
    plt.title('blended RGB Image')
    plt.axis('off')  # Turn off axis labels
    plt.show()
    return


def blend_hybrid_image(laplacian_pyramid_image1, laplacian_pyramid_image2):
    num_levels = min(len(laplacian_pyramid_image1), len(laplacian_pyramid_image2))
    hybrid_pyramid = []
    for i in range(num_levels):
        alpha = i / (num_levels - 1)  # Linearly interpolate between the two images
        hybrid_level = (1 - alpha) * laplacian_pyramid_image1[i] + alpha * laplacian_pyramid_image2[i]
        hybrid_pyramid.append(hybrid_level)
    return hybrid_pyramid


def hybrid_image(path_image_1, path_image_2):
    image1 = cv2.imread(path_image_1)
    image2 = cv2.imread(path_image_2)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)/255
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)/255
    laplacian_pyramid_image1 = laplacian_pyramid(gray_image1, 5)
    laplacian_pyramid_image2 = laplacian_pyramid(gray_image2, 5)
    hybrid_pyramid = blend_hybrid_image(laplacian_pyramid_image1, laplacian_pyramid_image2)
    image = sum_pyramid(hybrid_pyramid)
    plt.imshow(image, cmap='gray')
    plt.title('Hybrid Image')
    plt.axis('off')  # Turn off axis labels
    plt.show()
    return


