""" Kalman filtering """

import numpy as np
from filterpy.kalman import KalmanFilter as kalmanfilter


class KalmanFilter:
    def __init__(self, car, variance, width):
        """
        Constructs a kalman filter that tracks the car's position and velocity from GPS measurements
        car: the car being tracked (use car.pos and car.vel for the initial state estimate)
        variance: variance of gaussian GPS noise that the filter assumes (per axis)
        width: width of uniform GPS noise that the filter assumes (per axis the noise is uniform on [-width/2, width/2])
        You need to implement state transition matrix, measurement matrix, measurement noise covariance,
        process noise covariance, initial state estimate, and initial covariance matrix

        self.kf (created below) is a FilterPy KalmanFilter. Its state is the column vector
        [x, y, velocity_x, velocity_y] and its measurement is [x, y]. One call of predict_and_update
        is one time step (dt = 1). Set these attributes of self.kf (numpy arrays of floats):

            self.kf.F  (4 x 4)  state transition matrix: the state after one time step is F times the state
            self.kf.H  (2 x 4)  measurement matrix: the expected measurement is H times the state
            self.kf.R  (2 x 2)  measurement noise covariance: the variance of the GPS noise on each axis
                                (the x and y noise are independent)
            self.kf.Q  (4 x 4)  process noise covariance: how much the true motion can differ from F
                                in one step (your choice; you explain it in Task 3.2)
            self.kf.x  (4 x 1)  initial state estimate, from car.pos and car.vel
            self.kf.P  (4 x 4)  initial covariance: how uncertain the initial state estimate is (your choice)
        """
        self.kf = kalmanfilter(dim_x=4, dim_z=2)  # 4 states (x, y, velocity_x, velocity_y), 2 measurements (x, y)
        self.variance = variance
        self.width = width
        self.car = car

        # BEGIN_YOUR_CODE ######################################################
        raise NotImplementedError

        # END_YOUR_CODE ########################################################

    def predict_and_update(self, measurement, which="gaussian"):
        """
        Returns the state after predicting and updating
        measurement: GPS measurement, numpy array of shape (2,)
        which: "gaussian" or "uniform", the kind of GPS noise; set the measurement noise covariance
               from self.variance (gaussian) or self.width (uniform) before the update

        Three steps:
          1. Set self.kf.R for this kind of noise. For "gaussian", the GPS noise on each axis has
             variance self.variance. For "uniform", it is uniform on [-self.width/2, self.width/2];
             use the variance of that distribution.
          2. Predict:  self.kf.predict()
          3. Update with the measurement:  self.kf.update(z), where z is the measurement as a
             numpy array of shape (2, 1)
        self.kf.x then holds the new state estimate, of shape (4, 1).
        """
        # BEGIN_YOUR_CODE ######################################################
        raise NotImplementedError

        # END_YOUR_CODE ########################################################

        return self.kf.x
