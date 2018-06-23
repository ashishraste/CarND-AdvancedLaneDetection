import vision_util as vu
import numpy as np
import cv2

# ROI_BOTTOM_LEFT = (190, 720)
# ROI_TOP_LEFT = (585, 455)
# ROI_TOP_RIGHT = (705, 455)
# ROI_BOTTOM_RIGHT = (1130, 720)

sobelx_thresh = (20, 230)
hls_saturation_thresh = (180, 255)

def detection_pipeline(src_img):
    # Get Sobel(x)-filtered binary image.
    sobelx_binary = vu.abs_sobel_thresh(src_img, orient='x', thresh=sobelx_thresh)
    # Get Saturation-channel-thresholded binary image.
    hls_binary = vu.hls_select(src_img, thresh=hls_saturation_thresh)
    # Combined binary image having color and gradient thresholds applied.
    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(sobelx_binary == 1) | (hls_binary == 1)] = 1
    vu.plot_transformed_image(src_img, combined_binary, 'Filtered Image', gray_cmap=True)

# Calibrate camera to undistort images.
def calibration_pipeline(src_img):
    mtx, dist = vu.compute_calibration_matrix()
    vu.save_calibration_matrix(mtx, dist)
    # distort_test = vu.load_image('../camera_cal/calibration1.jpg')


def perspective_transform_pipeline(mtx, dist):
    # Finding perpective transformation matrix to get bird's eye view on images.
    src_image = vu.load_image('../test_images/straight_lines1.jpg')
    undistorted = vu.undistort_image(src_image, mtx, dist)
    return vu.find_perspective_transform(undistorted)

if __name__ == '__main__':
    mtx, dist = vu.load_calibration_matrix()
    ptrans_mat = perspective_transform_pipeline(mtx, dist)



