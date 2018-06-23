import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

def compute_calibration_matrix(image_dir='../camera_cal/calibration*.jpg', nx=9, ny=6):
    # Prepare object points.
    objp = np.zeros((ny*nx, 3), np.float32)
    # objp.reshape(-1,1,3)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    objpoints = []  # 3D points in the real world space.
    imgpoints = []  # 2D points in the image plane.

    image_dir = glob.glob(image_dir)
    # Step through the files and search for chessboard corners.
    for idx, fname in enumerate(image_dir):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if ret is True:
            # If found, add object points and their respective corners (image points).
            objpoints.append(objp)
            imgpoints.append(corners)

    # Find calibration matrix.
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

def save_calibration_matrix(matrix, dist):
    calib_details = {'calibration_matrix':matrix, 'distortion_coefficients':dist}
    with open('./calibration.p', 'wb') as calib_file:
        pickle.dump(calib_details, calib_file)

def load_calibration_matrix(calib_path = './calibration.p'):
    calib_details = pickle.load(open(calib_path, 'rb'))
    return calib_details['calibration_matrix'], calib_details['distortion_coefficients']

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  if orient == 'x':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  if orient == 'y':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  abs_sobel = np.absolute(sobel)
  scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
  return binary_output

def hls_select(img, thresh=(0, 255)):
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2]
  binary_output = np.zeros_like(s_channel)
  binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
  return binary_output

def draw_roi(img, left_line_endpoints, right_line_endpoints, color=[255, 0, 0], thickness=1):
  """
  Draws the region-of-interest box on an image.
  :param img: Source image on which the ROI has to be drawn.
  :param left_line_endpoints: Endpoints of the line-segment belonging to the left line of the box.
  :param right_line_endpoints: Endpoints os the line-segment belonging to the right line of the box.
  :param color: RGB color of the line to be drawn.
  :param thickness: Thickness of the line to be drawn.
  """
  # Draw left, top, right and bottom lines of the rectangular ROI.
  cv2.line(img, left_line_endpoints[0], left_line_endpoints[-1], color, thickness)
  cv2.line(img, left_line_endpoints[-1], right_line_endpoints[-1], color, thickness)
  cv2.line(img, right_line_endpoints[0], right_line_endpoints[-1], color, thickness)
  cv2.line(img, left_line_endpoints[0], right_line_endpoints[0], color, thickness)
  return img

def find_perspective_transform(src_img):
    # Constants.
    roi_bottom_left = (200, 720)
    roi_top_left = (560, 475)
    roi_top_right = (730, 475)
    roi_bottom_right = (1110, 720)

    drawn_lines_img = draw_roi(
        src_img, (roi_bottom_left, roi_top_left), (roi_bottom_right, roi_top_right))
    img_size = (drawn_lines_img.shape[1], drawn_lines_img.shape[0])

    # Pick outermost corners for unwarping the image.
    outer_rect_corners = [[roi_bottom_left[0], roi_bottom_left[1]],
                          [roi_top_left[0], roi_top_left[1]],
                          [roi_top_right[0], roi_top_right[1]],
                          [roi_bottom_right[0], roi_bottom_right[1]]]
    outer_rect_corners = np.array(outer_rect_corners, np.float32)
    outer_rect_corners = outer_rect_corners.reshape(-1, outer_rect_corners.shape[-1])

    # Pick destination corners arbitrarily where the source image's corners can be mapped to.
    offset = 100
    dst_rect_corners = [[roi_bottom_left[0] + offset, roi_bottom_left[1]],
                        [roi_bottom_left[0] + offset, 0],
                        [roi_bottom_right[0] - offset, 0],
                        [roi_bottom_right[0] - offset, roi_bottom_right[1]]]
    dst_rect_corners = np.array(dst_rect_corners, np.float32)
    dst_rect_corners = dst_rect_corners.reshape(-1, dst_rect_corners.shape[-1])
    ptrans_mat = cv2.getPerspectiveTransform(outer_rect_corners, dst_rect_corners)
    # warped = cv2.warpPerspective(drawn_lines_img, ptrans_mat, img_size)
    # plot_transformed_image(drawn_lines_img, warped, 'Image with source points', 'Warped image', save_result=True)
    return ptrans_mat

def plot_transformed_image(
    src, dst, source_image_title='Original Image', transformed_img_title='Undistorted Image', gray_cmap=False,
    axis_off=False, save_result=False):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.set_title(source_image_title, fontsize=50)
    ax2.set_title(transformed_img_title, fontsize=50)
    if axis_off is True:
        ax1.axis('off')
        ax2.axis('off')
    if gray_cmap is True:
        ax1.imshow(src, cmap='gray')
        ax2.imshow(dst, cmap='gray')
    else:
        ax1.imshow(src)
        ax2.imshow(dst)
    if save_result is True:
        plt.imsave('../output_images/'+transformed_img_title, dst)
    plt.show()

def warp(src_img, ptrans_mat):
    img_size = (src_img.shape[1], src_img.shape[0])
    return cv2.warpPerspective(src_img, ptrans_mat, img_size)

def undistort_image(src_image, mat, dist):
    return cv2.undistort(src_image, mat, dist, None, mat)

def load_image(img_path = '../camera_cal/calibration1.jpg'):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def save_image(img, img_path = '../output_images/undistorted_chessboard.jpg'):
    plt.imsave(img_path, img)
