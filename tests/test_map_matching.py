"""Public tests for Part 4 (map_matching.py)."""

import itertools
import time

import numpy as np

import track_hmm as H
from map_matching import transition_model, sensor_likelihood, forward, viterbi

PART = "4: map matching (HMM, Viterbi)"


def points(n, part):
    def deco(fn):
        fn.points, fn.part = n, part
        return fn
    return deco


def seeds(base, per_case=1):
    return [base + i for i in range(per_case)]


def close(a, b, tol=1e-8, msg=""):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "{} shape {} != expected {}".format(msg, a.shape, b.shape)
    assert np.allclose(a, b, atol=tol, rtol=tol), "{}\n got      {}\n expected {}".format(msg, np.round(a, 5), np.round(b, 5))


def random_model(rng, S, t, sparsity=0.0):
    """a random HMM with S states and random likelihoods for t readings (no ties)"""
    T = rng.random((S, S)) * (rng.random((S, S)) >= sparsity)
    T[np.arange(S), rng.integers(S, size=S)] += 0.1
    T /= T.sum(1, keepdims=True)
    L = rng.random((t, S)) + 0.01
    prior = rng.random(S) + 0.1
    return prior / prior.sum(), T, L


def all_sequences(prior, T, L, steps):
    """every state sequence for the first `steps` readings, with its joint probability with those readings"""
    S = L.shape[1]
    first = prior @ T
    for path in itertools.product(range(S), repeat=steps):
        p = first[path[0]] * L[0, path[0]]
        for k in range(1, steps):
            p *= T[path[k - 1], path[k]] * L[k, path[k]]
        yield path, p


def brute_force(prior, T, L):
    """filtered beliefs and the probability of the most likely sequence, by enumerating all sequences"""
    t, S = L.shape
    filt = np.zeros((t, S))
    best = 0.0
    for k in range(t):
        for path, p in all_sequences(prior, T, L, k + 1):
            filt[k, path[-1]] += p
            if k == t - 1:
                best = max(best, p)
        filt[k] /= filt[k].sum()
    return filt, best


def path_log_prob(prior, T, L, path):
    with np.errstate(divide="ignore"):
        lp = np.log((prior @ T)[path[0]]) + np.log(L[0, path[0]])
        for k in range(1, len(path)):
            lp += np.log(T[path[k - 1], path[k]]) + np.log(L[k, path[k]])
    return float(lp)


@points(2, PART)
def test_transition_model():
    for S, probs in ((6, (0.2, 0.5, 0.3)), (5, (0.6, 0.4)), (7, (0.1, 0.2, 0.3, 0.4))):
        T = np.asarray(transition_model(S, probs), float)
        assert T.shape == (S, S)
        close(T.sum(axis=1), np.ones(S), msg="rows of T must sum to 1")
        for d, p in enumerate(probs):
            close(T[1, 1 + d], p, msg="S={}: moving {} points forward from point 1".format(S, d))
        close(T[S - 1, (len(probs) - 2) % S], probs[-1], msg="S={}: after the last point comes point 0".format(S))
        close(T[2, 1], 0.0, msg="the car never moves backward")


@points(2, PART)
def test_sensor_likelihood():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 100.0]])
    z = np.array([[4.0, -5.0], [12.0, 3.0]])
    L = np.asarray(sensor_likelihood(z, pts, "uniform", width=10.0), float)
    assert L.shape == (2, 3)
    close(L, [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], msg="uniform, width 10 (|error| <= 5 on each axis)")
    L = np.asarray(sensor_likelihood(z, pts, "gaussian", var=9.0), float)
    assert L.shape == (2, 3)
    close(L[0, 0], 0.0018128059, tol=1e-9, msg="gaussian, var 9, reading 0, state 0")
    close(L[1, 1], 0.0085885628, tol=1e-9, msg="gaussian, var 9, reading 1, state 1")


@points(1, PART)
def test_forward():
    for seed in seeds(0, 4):
        rng = np.random.default_rng(seed)
        prior, T, L = random_model(rng, 3, 6, sparsity=0.3)
        close(forward(prior, T, L), brute_force(prior, T, L)[0], msg="seed {}".format(seed))
    rng = np.random.default_rng(seeds(50)[0])
    prior, T, L = random_model(rng, 10, 3000)
    b = forward(prior, T, L * 1e-3)
    assert np.all(np.isfinite(b)), "normalize the forward message at every step"
    close(b.sum(axis=1), np.ones(3000))


@points(2, PART)
def test_viterbi_small():
    for seed in seeds(100, 6):
        rng = np.random.default_rng(seed)
        prior, T, L = random_model(rng, 3, 6, sparsity=0.3)
        path = np.asarray(viterbi(prior, T, L))
        assert path.shape == (6,)
        close(path_log_prob(prior, T, L, path), np.log(brute_force(prior, T, L)[1]),
              msg="seed {}: your path is not a most likely sequence".format(seed))


@points(1, PART)
def test_viterbi_long():
    rng = np.random.default_rng(seeds(70)[0])
    prior, T, L = random_model(rng, 200, 400, sparsity=0.9)
    L = L * 1e-3
    t0 = time.perf_counter()
    path = np.asarray(viterbi(prior, T, L))
    assert time.perf_counter() - t0 < 20, "too slow"
    assert path.shape == (400,)
    lp = path_log_prob(prior, T, L, path)
    assert np.isfinite(lp), "the path has probability 0: divide the message by its maximum at every step"


@points(1, PART)
def test_viterbi_racing_line():
    pts = H.racing_line()
    S = len(pts)
    T = np.asarray(transition_model(S, H.STEP_PROBS), float)
    prior = H.uniform_prior(S)
    for seed in seeds(30):
        d = H.simulate_drive(seed, pts, "gaussian", var=400.0, steps=120)
        L = np.asarray(sensor_likelihood(d["readings"], pts, "gaussian", var=400.0), float)
        path = np.asarray(viterbi(prior, T, L))
        assert H.impossible_moves(path, S) == 0, "drive {}: the Viterbi path makes an impossible move".format(seed)
        assert path_log_prob(prior, T, L, path) >= path_log_prob(prior, T, L, d["states"]) - 1e-6, \
            "drive {}: the true sequence of states is more likely than your path".format(seed)
