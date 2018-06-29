import vision_util as vu
from line import Line
from moviepy.editor import VideoFileClip
import logging

# Logger setup.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals.
# Matrices and camera distortion coefficients pickled for reuse.
mtx, dist = vu.load_calibration_matrix()
ptrans_mat, ptrans_mat_inv = vu.load_perspective_transform_matrix()

window_size = 8  # Number of frames to be used for detected-lane-smoothing.
left_line = Line(window_size)
right_line = Line(window_size)
lane_detected = False  # Whether a lane was detected in a new image-frame using previously detected lane(s).
curv_left, curv_right = 0.0, 0.0  # Radius of curvature for left and right lane-lines.


def calibration_pipeline():
    """
    # Calibrates camera to undistort images.
    """
    mtx, dist = vu.compute_calibration_matrix()
    vu.save_calibration_matrix(mtx, dist)


def warp_pipeline(src_img, ptrans_mat):
    """
    Applies thresholding (color/gradient) and perspective transformation on an undistorted image.
    :param src_img: Undistorted image.
    :param ptrans_mat: Perspective transformation matrix
    :return: Warped binary which is aa filtered top-down version of the original image.
    """
    binary = vu.combined_threshold(src_img)
    binary_warped = vu.warp(binary, ptrans_mat)
    return binary_warped


def lane_detection_pipeline(src_img):
    """
    Detects the lane in an image-frame and annotates it with curvature and vehicle-offset details.
    1. Undistort image.
    2. Apply warp-pipeline to get binary_warped image.
    3. Initialize lane-lines .
    4. Detect lane-lines if not detected previously, else update lane-lines.
    """
    global mtx, dist, ptrans_mat, ptrans_mat_inv
    global lane_detected, curv_left, curv_right
    undistorted = vu.undistort_image(src_img, mtx, dist)
    binary_warped = warp_pipeline(undistorted, ptrans_mat)

    if lane_detected is False:
        # Detect lane by going through the image in parts and fitting polynomials.
        lane_lines_img, left_fit, right_fit = vu.detect_lane_lines(binary_warped)

        # Add detected lane-lines to existing moving average and update them.
        left_fit = left_line.update_fit(left_fit)
        right_fit = right_line.update_fit(right_fit)

        # Compute curvature of newly detected lane-lines.
        curv_left, curv_right = vu.calculate_curvature(src_img, left_fit, right_fit)
        # Guaranteed to identify lane lines, if any.
        lane_detected = True

    else:
        # Highly targeted search for lane-lines based on previously detected lane-lines.
        left_fit = left_line.get_fit()
        right_fit = right_line.get_fit()
        ret = vu.update_detected_lane(binary_warped, left_fit, right_fit)

        if ret is not None:
            # Updated left and right lines' were detected.
            updated_left_fit = ret['left_fit']
            updated_right_fit = ret['right_fit']

            # Update the moving-average.
            left_fit = left_line.update_fit(updated_left_fit)
            right_fit = right_line.update_fit(updated_right_fit)

            # Compute curvature.
            curv_left, curv_right = vu.calculate_curvature(src_img, left_fit, right_fit)

        else:  # Potential problem. New detections and polynomials have to be found across the image.
            logger.warning("Lane lost!")
            lane_detected = False

    vehicle_offset = vu.calculate_vehicle_offset(src_img, left_fit, right_fit)
    return vu.draw_lane(src_img, left_fit, right_fit, ptrans_mat_inv, curv_left, curv_right, vehicle_offset)


def detection_pipeline_test():
    """
    Sanity check for the lane-detection pipeline.
    """
    img = vu.load_image('../test_images/test3.jpg')
    detected_lane = lane_detection_pipeline(img)
    vu.plot_transformed_image(img, detected_lane, 'Original Image', 'Detected Lane Image', axis_off=True)
    # detected_lane = lane_detection_pipeline(img)
    # vu.plot_transformed_image(img, detected_lane, 'Original Image', 'Detected Lane Image', axis_off=True)
    # Failure case.
    # img = vu.load_image('../test_images/shadow_failure.jpg')
    # detected_lane = lane_detection_pipeline(img)
    # vu.plot_transformed_image(img, detected_lane, 'Original Image', 'Detected Lane Image', axis_off=True)


def run_video_pipeline(input_video_file, output_video_file):
    """
    Runs lane-detection pipeline on an input video and
    saves the annotated version in an another video.
    """
    input = VideoFileClip(input_video_file)
    detected_lane_video = input.fl_image(lane_detection_pipeline)
    detected_lane_video.write_videofile(output_video_file, audio=False)


if __name__ == '__main__':
    # detection_pipeline_test()  # Sanity check for the detection pipeline.
    run_video_pipeline('../project_video.mp4', '../detected_lane.mp4')