import numpy as np

class Lane():
    def __init__(self):
        self.left = Line()
        self.right = Line()
        self.midpoint = None

    def update(self, binary_warped):
        if self.midpoint == None or self.left.detected == False or self.right.detected == False:
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
            self.midpoint = np.int(histogram.shape[0]//2)
            self.left.line_base_pos = np.argmax(histogram[:self.midpoint])
            self.right.line_base_pos = self.midpoint + np.argmax(histogram[self.midpoint:])
            num_windows = 9
            window_height = np.int(binary_warped.shape[0] // num_windows)
        pass


# Class encapsulating a detected line.
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None