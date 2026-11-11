"""Public tests for Part 3 (kalman_filter.py)."""

import numpy as np

from kalman_filter import KalmanFilter

PART = "3: Kalman filter"


def points(n, part):
    def deco(fn):
        fn.points, fn.part = n, part
        return fn
    return deco


def seeds(base, per_case=1):
    return [base + i for i in range(per_case)]


class FakeCar:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)


def straight_drive(seed, steps=120, noise="gaussian", var=25.0, width=20.0):
    rng = np.random.default_rng(seed)
    p0, v = rng.uniform(100, 500, 2), rng.uniform(-6, 6, 2)
    truth = np.array([p0 + v * (k + 1) for k in range(steps)])
    if noise == "gaussian":
        z = truth + rng.normal(0, np.sqrt(var), truth.shape)
    else:
        z = truth + rng.uniform(-width / 2, width / 2, truth.shape)
    return p0, v, truth, z


@points(2, PART)
def test_model_matrices():
    kf = KalmanFilter(FakeCar([10, 20], [1, 2]), 9.0, 30.0)
    assert np.asarray(kf.kf.F, float).shape == (4, 4), "F must be 4 x 4 (state x, y, vx, vy)"
    assert np.asarray(kf.kf.H, float).shape == (2, 4), "H must be 2 x 4 (the GPS gives x, y)"
    assert np.allclose(np.ravel(kf.kf.x)[:2], [10, 20]), "start the state estimate at the car's position"
    for which in ("gaussian", "uniform"):
        kf.predict_and_update(np.array([10.0, 20.0]), which)
        R = np.asarray(kf.kf.R, float)
        assert R.shape == (2, 2) and np.allclose(R, R.T) and np.all(np.linalg.eigvalsh(R) > 0), \
            "R ({} GPS noise) must be a covariance matrix".format(which)
    Q = np.asarray(kf.kf.Q, float)
    assert Q.shape == (4, 4) and np.allclose(Q, Q.T) and np.all(np.linalg.eigvalsh(Q) >= -1e-12), "Q: a covariance matrix"


@points(2, PART)
def test_tracks_constant_velocity():
    for s in seeds(0):
        p0, v, truth, _ = straight_drive(s)
        kf = KalmanFilter(FakeCar(p0, [0.0, 0.0]), 25.0, 20.0)
        for k in range(len(truth)):
            x = np.ravel(kf.predict_and_update(truth[k], "gaussian"))
        assert np.linalg.norm(x[:2] - truth[-1]) < 1.0, "the estimate must follow noiseless measurements"
        assert np.linalg.norm(x[2:4] - v) < 0.3, "the velocity must be estimated from the measurements"


@points(2, PART)
def test_reduces_gps_noise():
    for s in seeds(10, 2):
        for noise in ("gaussian", "uniform"):
            p0, v, truth, z = straight_drive(s, noise=noise, var=100.0, width=40.0)
            kf = KalmanFilter(FakeCar(p0, v), 100.0, 40.0)
            est = np.array([np.ravel(kf.predict_and_update(z[k], noise))[:2] for k in range(len(z))])
            gps = np.sqrt(np.mean(np.sum((z - truth) ** 2, axis=1)))
            filt = np.sqrt(np.mean(np.sum((est[20:] - truth[20:]) ** 2, axis=1)))
            assert filt < 0.85 * gps, "{} noise: filter RMS error {:.2f} vs GPS {:.2f}".format(noise, filt, gps)
