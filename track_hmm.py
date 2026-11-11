"""
GPS map matching on the racing line (Part 4): the model and simulated drives.

The racing line of the recorded lap (data/lap_data.npy) is cut into points SPACING pixels apart.
The hidden state X_t is the index of the point where the car is at step t; the points form a loop,
so after the last point comes point 0. In each step the car stays at its point or moves 1 or 2
points forward, with probabilities STEP_PROBS; it never moves backward. The evidence E_t is a GPS
reading: the car's (x, y) position plus noise on each axis.

The noise of the GPS is the same as in Part 3, independent on the x and y axes:
    "uniform":  uniform on [-width/2, width/2] on each axis
    "gaussian": Gaussian with mean 0 and variance var on each axis

You do not need to change this file.
"""

import numpy as np

SPACING = 15.0               # pixels between two points of the racing line
STEP_PROBS = (0.2, 0.5, 0.3)  # P(move 0, 1, or 2 points forward in one step)
STEPS = 200                  # steps per drive


def racing_line(path="data/lap_data.npy", spacing=SPACING):
    """
    returns points, numpy array of shape (S, 2): the (x, y) positions of the S states, in driving order.
    The lap is cut where it returns to its start, so the points form a closed loop.
    """
    lap = np.load(path)[:, :2]
    seg = np.linalg.norm(np.diff(lap, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    half = len(lap) // 2
    end = half + int(np.argmin(np.linalg.norm(lap[half:] - lap[0], axis=1)))
    ss = np.arange(0.0, s[end], spacing)
    return np.c_[np.interp(ss, s, lap[:, 0]), np.interp(ss, s, lap[:, 1])]


def gps_noise(rng, n, noise_dist="uniform", width=60.0, var=300.0):
    """n samples of the 2-D GPS error"""
    if noise_dist == "uniform":
        return rng.uniform(-width / 2.0, width / 2.0, size=(n, 2))
    if noise_dist == "gaussian":
        return rng.normal(0.0, np.sqrt(var), size=(n, 2))
    raise ValueError("unknown noise distribution " + str(noise_dist))


def simulate_drive(seed, points, noise_dist="uniform", width=60.0, var=300.0, steps=STEPS):
    """
    Simulates one drive of `steps` steps from a random starting point.
    returns dict with
        states:   numpy int array (steps,), the true state X_1, ..., X_steps
        readings: numpy array (steps, 2), the GPS readings E_1, ..., E_steps
        true_xy:  numpy array (steps, 2), the true positions points[states]
    """
    rng = np.random.default_rng(20000 + seed)
    S = len(points)
    x = int(rng.integers(S))
    states = np.zeros(steps, dtype=int)
    for t in range(steps):
        x = (x + int(rng.choice(len(STEP_PROBS), p=STEP_PROBS))) % S
        states[t] = x
    true_xy = points[states]
    readings = true_xy + gps_noise(rng, steps, noise_dist, width, var)
    return dict(states=states, readings=readings, true_xy=true_xy)


def uniform_prior(S):
    """P(X_0): the car may be anywhere on the racing line"""
    return np.full(S, 1.0 / S)


def first_impossible_step(prior, T, L):
    """
    Checks whether the readings are possible under the model.
    L: numpy array (t, S), L[k, j] = P(E_{k+1} | X_{k+1} = j) under the ASSUMED sensor model.
    returns the first step k (0-based) at which every state sequence has probability 0, or None.
    """
    possible = np.asarray(prior) > 0            # states the car can be in, given the readings so far
    for k in range(len(L)):
        reachable = (possible.astype(float) @ (np.asarray(T) > 0)) > 0
        possible = reachable & (np.asarray(L[k]) > 0)
        if not possible.any():
            return k
    return None


def impossible_moves(path, S, max_step=len(STEP_PROBS) - 1):
    """number of steps in which a sequence of states moves backward or more than max_step points forward"""
    d = np.mod(np.diff(path), S)
    return int(np.sum(d > max_step))


def position_error(path, points, true_xy):
    """distance in pixels between the estimated positions points[path] and the true positions"""
    return np.linalg.norm(points[np.asarray(path)] - true_xy, axis=1)
