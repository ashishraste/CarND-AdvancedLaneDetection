import numpy as np

class Line():
    """
    Class encapsulating a detected lane-line.
    """
    def __init__(self, window_size):
        """
        Class constructor.
        :param window_size: Size of the moving average window holding lane-line polynomial coefficients.
        Lane line's second order polynomial equation being f(y) = Ay^2 + By + C
        """
        self.window_size = window_size
        # List holding the coefficients of detected polynomial equations.
        self.poly_coeffs_list = []
        # Mean average of coefficients across the lines observed.
        self.avg_line_fit = np.zeros(shape=(1,3), dtype=np.float64)

    def update_fit(self, line_fit):
        """
        Adds a new poly-line's coefficients and updates the moving-average.
        :param line_fit: Current polynomial fit over previously detected line.
        :return: Updated polynomial fit over currently detected line,
        taking into consideration previous polynomial-fits.
        """
        # Ensure line_fit is a single row with its coefficients on columns.
        # TODO: Test using the tuple as such instead of converting to/from numpy-array.
        line_fit = line_fit.T
        self.poly_coeffs_list.append(line_fit)
        if len(self.poly_coeffs_list) >= self.window_size:
            _ = self.poly_coeffs_list.pop(0)
        self.avg_line_fit = np.mean(self.poly_coeffs_list, axis=0)
        return self.avg_line_fit.T.reshape(-1,)

    def get_fit(self):
        """
        Gets the current (averaged) line-fit.
        """
        return self.avg_line_fit.T.reshape(-1,)