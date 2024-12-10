#!/usr/bin/env python3

"""
Implement Constant Velocity Kalman Filter class.
The variables we want to estimate are x and y positions, yaw angle of the bounding box,
and bbox dimensions. For this we assume, a constant velocity regarding x and y and a constant
turn rate for yaw. We assume constant width and height.

Our 8 state variables:
- x
- y
- yaw
- speed along x (vx)
- speed along y (vy)
- turn rate (d_yaw/dt)
- width
- height

Our 5 measurements:
- x
- y
- yaw
- width
- height
"""

import numpy as np

TIME_INTERVAL = 1.0 / 10.0


def adjust(meas, theta):
    """
    Find closest yaw angle to theta, that equals meas (modulo 2 PI)
    """
    assert (meas - theta) % (2 * np.pi) >= 0
    adjusted = meas
    if (meas - theta) % (2 * np.pi) > np.pi:
        adjusted = (meas - theta) % (2 * np.pi) - 2 * np.pi + theta
    else:
        adjusted = (meas - theta) % (2 * np.pi) + theta

    return adjusted


class KalmanFilter:
    """
    Kalman Filter with CV model
    Define state, covariance, predict & update steps
    """

    def __init__(self, nx, nz, first_measurement):
        self.nx = nx  # x, y, yaw, vx, vy, tr, w, h
        self.nz = nz  # x, y, yaw, w, h
        self.K = np.zeros((nx, nz))
        assert first_measurement.shape == (nz,)

        self.estimate = np.array(
            [
                first_measurement[0],  # x
                first_measurement[1],  # y
                first_measurement[2],  # yaw
                0,  # vx
                0,  # vy
                0,  # tr
                first_measurement[3],  # w
                first_measurement[4],  # h
            ]
        ).reshape(8, 1)

        # Initial covariance estimate
        self.P = np.diag(
            [
                100**2,
                100**2,
                (30 * np.pi / 180) ** 2,
                500**2,
                500**2,
                (30 / TIME_INTERVAL * np.pi / 180) ** 2,
                5**2,
                5**2,
            ]
        )
        assert np.all(self.P.T == self.P)

        # Jacobian measurement matrix
        self.H = np.zeros((nz, nx))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 6] = 1
        self.H[4, 7] = 1

        # Measurement noise R
        XY_STD_MEASUREMENT = 5  # 5 pixels accuracy for x/y position
        YAW_STD_MEASUREMENT = 30 * np.pi / 180  # 30 degrees accuracy
        WH_STD_MEASUREMENT = 10  # 10 pixels std for width/height
        self.R = np.diag(
            [
                XY_STD_MEASUREMENT**2,
                XY_STD_MEASUREMENT**2,
                YAW_STD_MEASUREMENT**2,
                WH_STD_MEASUREMENT**2,
                WH_STD_MEASUREMENT**2,
            ]
        )  # measurement uncertainty

        self.I = np.identity(nx)

        # Process noise (the most touchy part)
        # values too low --> lag error
        # values too high --> KF follows measurements and we have noisy estimation
        # It s possible that you change these values for your current situation
        self.Q = np.zeros((nx, nx))
        MAX_SPEED = 5
        MAX_TR = 3
        self.Q[0, 0] = (TIME_INTERVAL * MAX_SPEED) ** 2
        self.Q[1, 1] = (TIME_INTERVAL * MAX_SPEED) ** 2
        self.Q[2, 2] = (TIME_INTERVAL * MAX_TR) ** 2
        self.Q[3, 3] = MAX_SPEED**2
        self.Q[4, 4] = MAX_SPEED**2
        self.Q[5, 5] = MAX_TR**2
        self.Q[6, 6] = 1**2
        self.Q[7, 7] = 1**2

        # Model matrix (constant velocity and turn rate)
        self.F = np.zeros((nx, nx))
        self.F[0, 0] = 1
        self.F[1, 1] = 1
        self.F[2, 2] = 1
        self.F[3, 3] = 1
        self.F[4, 4] = 1
        self.F[5, 5] = 1
        self.F[0, 3] = TIME_INTERVAL
        self.F[1, 4] = TIME_INTERVAL
        self.F[2, 5] = TIME_INTERVAL
        self.F[6, 6] = 1
        self.F[7, 7] = 1

        assert np.all(self.Q.T == self.Q)

        assert self.estimate.shape == (nx, 1)
        assert self.P.shape == (nx, nx)
        assert self.H.shape == (nz, nx)
        assert self.I.shape == (nx, nx)
        assert self.R.shape == (nz, nz)
        assert self.Q.shape == (nx, nx)
        assert self.K.shape == (nx, nz)

    def predict(self):
        """
        Predict state and covariance (linear)
        """

        self.estimate = self.F.dot(self.estimate)  # no control on the system

        # Predict covariance
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, measurement):
        """
        Update step
        Update Kalman gain K
        Update state estimate
        Update state convariance P
        """
        # Update Kalman Gain

        assert measurement.shape == (self.nz,)

        # Linearization around current estimate
        estimate = self.estimate.reshape(self.nx, 1)

        # Innovation
        S = self.H.dot(self.P).dot(self.H.T) + self.R

        # Kalman gain
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # assert that this was invertible

        assert self.estimate.shape == (self.nx, 1)
        assert self.K.shape == (self.nx, self.nz)

        measurement[2] = adjust(measurement[2], estimate[2, 0])

        # State update equation
        diff = measurement.reshape(self.nz, 1) - self.H.dot(estimate)
        self.estimate = estimate + self.K.dot(diff)

        assert self.estimate.shape == (self.nx, 1)

        # Covariance update equation
        self.P = (self.I - self.K.dot(self.H)).dot(self.P)
