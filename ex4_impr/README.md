# Image Alignment and Blending using SIFT + RANSAC

This exercise aligns two images using **SIFT feature matching** and **homography estimation with RANSAC**, then performs seamless blending using an alpha mask.

## ✨ Features

- Feature extraction with SIFT
- Matching using FLANN + ratio test
- Homography estimation with RANSAC
- Warping and blending of images using alpha channel

## 📚 Related Topics

- Image alignment
- Geometric transformations
- Robust estimation (RANSAC)
- Alpha blending

## ▶️ Run

```bash
python ex4.py
```

Use `blend_image(path_low_image, path_high_image)` inside your script or interactively.

---

These methods are commonly used in tasks like:
- Panorama stitching
- Augmented reality
- Image registration
