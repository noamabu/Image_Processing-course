import cv2
import numpy as np


def blend_image(path_low_image, path_high_image):
    image1 = cv2.imread(path_low_image)
    image2 = cv2.imread(path_high_image, cv2.IMREAD_UNCHANGED)
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # SIFT descriptor extraction
    sift = cv2.SIFT_create()
    # Compute keypoints and descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)
    # Match descriptors between the two images using FLANN matcher
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    # Convert keypoints to numpy format
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Apply RANSAC to filter matches
    M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    # Warp image2 onto image1 using the transformation matrix M
    h, w = gray_image2.shape
    warped_image2 = cv2.warpPerspective(image2, M, (w, h))
    alpha_channel = warped_image2[:, :, 3]
    # Convert the alpha channel to a mask by thresholding
    _, mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    # Invert the mask (since OpenCV treats white as foreground and black as background)
    mask_inv = cv2.bitwise_not(mask)
    # Extract the RGB channels from image2
    image2_rgb = warped_image2[:, :, :3]
    # Black out the area of the first image where the warped image will be placed
    image1_bg = cv2.bitwise_and(image1, image1, mask=mask_inv)
    # Take only the region of the warped image that is not black
    warped_fg = cv2.bitwise_and(image2_rgb, image2_rgb, mask=mask)
    # Combine the images
    return cv2.add(image1_bg, warped_fg)

