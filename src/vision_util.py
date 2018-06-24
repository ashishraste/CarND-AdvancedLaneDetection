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
    src, dst, source_image_title, transformed_img_title, gray_cmap=False,
    axis_off=False, save_result=False):
    '''
    Plots source image and its transformed version.
    '''
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


def undistort_image(src_img, mat, dist):
    return cv2.undistort(src_img, mat, dist, None, mat)


def warp(src_img, ptrans_mat):
    img_size = (src_img.shape[1], src_img.shape[0])
    return cv2.warpPerspective(src_img, ptrans_mat, img_size)


def plot_histogram(binary_img):
    histogram = np.sum(binary_img[binary_img.shape[0]//2:,:], axis=0)
    # plt.plot(histogram)
    # plt.show()
    return histogram


def lane_detection_pipeline(binary_warped, visualize_lane=False):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = midpoint + np.argmax(histogram[midpoint:])

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    num_windows = 9
    window_height = np.int(binary_warped.shape[0] // num_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    min_pixels = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(num_windows):
        win_y_low = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_leftx_low = leftx_current - margin
        win_leftx_high = leftx_current + margin
        win_rightx_low = rightx_current - margin
        win_rightx_high = rightx_current + margin
        if visualize_lane:
            # Draw the windows on visualization image.
            cv2.rectangle(out_img, (win_leftx_low, win_y_low), (win_leftx_high, win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img, (win_rightx_low, win_y_low), (win_rightx_high, win_y_high), (0,255,0), 2)
        # Identify nonzero x and y pixels in the identified left/right windows.
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &\
            (nonzerox >= win_leftx_low) & (nonzerox < win_leftx_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &\
            (nonzerox >= win_rightx_low) & (nonzerox < win_rightx_high)).nonzero()[0]
        # Append good-indices to the lists.
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If greater minimum-pixels found, recenter next (left/right) windows to the mean of their location.
        if len(good_left_inds) > min_pixels:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pixels:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate all the arrays of indices found so far.
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right lines' pixel locations.
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fitting a second order polynomial over the left and right lines' pixels.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fitting for f(y) rather than f(x) since y varies (variable) and x could remain constant for different y values.
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if visualize_lane:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    return out_img, left_fit, right_fit


def calculate_curvature(src_img, left_fit, right_fit):
    '''
    Calculates curvature of a lane-line with the following equation.
    Curvature = (1 + (2Ay + B)^2)^(1.5) / |2A|, for a polynomial curve f(y) = Ay^2 + By + C.
    :param src_img: Source image containing lane-lines drawn.
    :param left_fit: Polynomial equation of degree two representing left lane-line.
    :param right_fit: Polynomial equation of degree two representing right lane-line.
    :return: Curvature of left and right lane-lines.
    '''
    ym_per_pix = 30 / src_img.shape[0]  # 30 metres lane-length w.r.t the camera top-down image.
    xm_per_pix = 3.7 / src_img.shape[1] # 3.7 metres lane-width w.r.t the camera top-down image.
    y_eval = src_img.shape[0] - 1
    # Converting coefficients of curvature-equations to metres.
    left_fit = np.array([(xm_per_pix/(ym_per_pix**2)) * left_fit[0], (xm_per_pix/ym_per_pix) * left_fit[1]])
    right_fit = np.array([(xm_per_pix/(ym_per_pix**2)) * right_fit[0], (xm_per_pix/ym_per_pix) * right_fit[1]])
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    return left_curverad, right_curverad


def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def save_image(img, img_path):
    plt.imsave(img_path, img)
