# Lane detection

Lane detection on roads using camera image-feed.

<img src="output_images/sample_detection.gif?raw=true">

## Overview
This project is part of [Udacity's Self-Driving Car Nanodegree program](https://www.udacity.com/drive)
and much of the source comes from the program's lecture notes and quizzes.

Following steps were performed to achieve a lane-detection pipeline which can
look for lanes in a video.

1. Camera calibration : Finds camera's calibration matrix and distortion coefficients.
2. Distortion correction : Un-distorts source images.
2. Image filtering : Thresholds an image containing a lane based on gradient and
color (HSV) values of its pixels.
3. Image warping : Perspective transformation is applied on the image to get a
"birds-eye" view of the filtered-image.
4. Lane detection and update : Lane's left and right lines are found and
their polynomial-fits representing them is found.
5. Computing lane-curvature and vehicle offset from lane-center.
6. Image warping : Projecting back detected lane on top-down view of the image to
the original image.
7. Writing the output image to an output-video.

## Dependencies
1. Python-3.5
2. OpenCV-Python
3. Moviepy
4. Numpy
5. Matplotlib
6. Pickle

## Running the detection routine
1. Switch to the source directory `src`:
```bash
cd src
```
2. Run the lane-detection script. This will take the video project_video.mp4 as its
input, run the detection pipeline on it and save the output to detected_lane.mp4 in the parent directory.
```bash
python lane_detection.py
```


## Directory Layout
* src : Contains the following source files.

|File  | Description |
|:----:|:-----------:|
|line.py | Line class capturing the characteristics of a lane-line. |
|vision_util.py | Visualization and detection utility. |
|lane_detection.py | Contains lane-detection pipeline with other (sub)pipelines like camera-calibration. |
| calibration.p | Pickled camera calibration matrix and distortion coefficients. |
| perspective_transform.p | Pickled perspective transform and its inverse transform matrices |

* camera_cal : Chessboard images for camera-calibration.
* test_images : Images of lanes which we used for testing the pipeline.
* output_images: Outputs from different parts of the pipeline which were applied
to images in *test_images*.
* detected_lane.mp4 : Output video containing the detected lane annotated with
lane-curvature and vehicle-offset from lane-center values.

## Detection pipeline description
Let's go through the detection routine steps mentioned above.

### Camera calibration and Distortion correction

### Image filtering

### Image warping for bird's eye view

### Lane detection and update

### Lane curvature and vehicle-offset

### Annotated source image with detected lanes

## Further improvements
