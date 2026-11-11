"""Public tests for Part 1 (particle_filter.py)."""

import random

import numpy as np

import particle_filter as pf
from particle_filter import Particle, ParticleFilter
from racetrack import load_racetrack
from utils import angle_bw

PART = "1: particle filter"


def points(n, part):
    def deco(fn):
        fn.points, fn.part = n, part
        return fn
    return deco


def seeds(base, per_case=1):
    """Seeds for a randomized test."""
    return [base + i for i in range(per_case)]


def seed_all(s):
    np.random.seed(s)
    random.seed(s)


def make_filter(n, **model):
    return ParticleFilter(n, 0, 1400, 0, 800, **model)


@points(2, PART)
def test_initialize_particles():
    for s in seeds(0):
        seed_all(s)
        f = make_filter(4000)
        ps = f.particles
        assert len(ps) == 4000 and all(isinstance(p, Particle) for p in ps)
        pos = np.array([p.pos for p in ps])
        ori = np.array([p.orient for p in ps])
        assert pos[:, 0].min() >= 0 and pos[:, 0].max() <= 1400 and pos[:, 1].min() >= 0 and pos[:, 1].max() <= 800
        assert abs(pos[:, 0].mean() - 700) < 30 and abs(pos[:, 1].mean() - 400) < 20, "positions must be uniform over the map"
        assert np.allclose(np.linalg.norm(ori, axis=1), 1.0), "orientations must be unit vectors"
        angles = np.arctan2(ori[:, 1], ori[:, 0])
        hist = np.histogram(angles, bins=8, range=(-np.pi, np.pi))[0]
        assert hist.min() > 350, "orientations must be uniform over all directions"


@points(3, PART)
def test_transition_sample():
    for s in seeds(10):
        seed_all(s)
        f = make_filter(1)
        rng = np.random.default_rng(s)
        a = rng.uniform(-np.pi, np.pi)
        p = Particle(np.array([500.0, 300.0]), np.array([np.cos(a), np.sin(a)]))
        delta, speed = rng.uniform(-0.3, 0.3), rng.uniform(2, 10)
        before = (p.pos.copy(), p.orient.copy())
        new = [f.transition_sample(p, delta, speed) for _ in range(2000)]
        assert np.allclose(p.pos, before[0]) and np.allclose(p.orient, before[1]), "do not change the particle passed in"
        assert all(isinstance(q, Particle) for q in new)
        pos = np.array([q.pos for q in new])
        ori = np.array([q.orient for q in new])
        expect_o = np.array([np.cos(a + delta), np.sin(a + delta)])
        expect_p = before[0] + speed * expect_o
        assert np.allclose(np.linalg.norm(ori, axis=1), 1.0)
        mean_o = ori.mean(axis=0) / np.linalg.norm(ori.mean(axis=0))
        assert np.linalg.norm(mean_o - expect_o) < 0.05, "rotate the orientation by delta_angle"
        assert np.linalg.norm(pos.mean(axis=0) - expect_p) < 1.5, "move speed units in the new direction"
        assert pos.std(axis=0).max() > 1e-3, "add noise to the new pose"
        # the rotation has the same sign as delta_angle in the simulator (angle_bw)
        assert abs(np.radians(angle_bw(expect_o, before[1])) - delta) < 1e-6


def fixed_sensor(r):
    return lambda x, y, m: np.array(r, dtype=float)


@points(4, PART)
def test_compute_prenorm_weight():
    p = Particle(np.array([100.0, 100.0]), np.array([1.0, 0.0]))
    r = [40.0, 60.0, 30.0, 80.0]
    z = np.array([42.0, 58.0, 30.5, 80.0])
    f = make_filter(1, noise_dist="gaussian", noise_std=3.0)
    assert np.isclose(f.compute_prenorm_weight(p, fixed_sensor(r), 200, z), 0.00019774413294183936, rtol=1e-9), \
        "Gaussian sensor model"
    f = make_filter(1, noise_dist="uniform", noise_bound=2.5)
    assert np.isclose(f.compute_prenorm_weight(p, fixed_sensor(r), 200, z), 0.0016), "uniform sensor model"
    z2 = np.array([43.0, 58.0, 30.5, 80.0])                       # one reading outside the bound
    assert f.compute_prenorm_weight(p, fixed_sensor(r), 200, z2) == 0.0, "a reading outside the bound has density 0"
    f = make_filter(1, noise_dist="uniform", noise_bound=2.5, fail_prob=0.1)
    assert np.isclose(f.compute_prenorm_weight(p, fixed_sensor(r), 200, z2), 2.9403675625e-06), \
        "random readings (fail_prob)"
    f = make_filter(1, noise_dist="gaussian", noise_std=3.0)
    calls = []

    def sensor(x, y, m):
        calls.append((x, y, m))
        return np.array(r)
    f.compute_prenorm_weight(p, sensor, 123, z)
    assert calls and np.allclose(calls[0][:2], [100, 100]) and calls[0][2] == 123, \
        "call sensor at the particle's position with max_sensor_range"


@points(2, PART)
def test_filtering_concentrates_on_matching_particles():
    for s in seeds(20):
        seed_all(s)
        f = make_filter(500, noise_dist="gaussian", noise_std=2.0)

        def sensor(x, y, m):                    # readings identify the left half of the map
            return np.array([0.0, 0.0, 0.0, 0.0]) if x < 700 else np.array([50.0, 50.0, 50.0, 50.0])
        new = f.filtering(sensor, 200, np.array([0.0, 0.0, 0.0, 0.0]), 0.0, 0.0)
        assert len(new) == 500 and all(isinstance(q, Particle) for q in new)
        xs = np.array([q.pos[0] for q in new])
        assert np.mean(xs < 705) > 0.97, "resampling must keep the particles that match the evidence"


def replay_errors(n, seed, steps=230):
    """Run car 1's recorded lap through a ParticleFilter with n particles; returns the position errors."""
    seed_all(seed)
    track = load_racetrack("data/racetrack.npz")
    lap = np.load("data/lap_data.npy")
    f = make_filter(n, noise_dist="gaussian", noise_std=10.0)
    errs = []
    for i in range(4, 4 + 2 * steps, 2):
        pos, prev, prev2 = lap[i, :2], lap[i - 2, :2], lap[i - 4, :2]
        vel, old = pos - prev, prev - prev2
        z = track.read_distances(pos[0], pos[1], 400, noisy=True, std=2.0)
        x, y, _ = f.filtering_and_estimation(track.read_distances, 400, z,
                                             np.radians(angle_bw(vel, old)), np.linalg.norm(vel))
        errs.append(np.hypot(x - pos[0], y - pos[1]))
    return np.array(errs)


@points(4, PART)
def test_localizes_car_on_recorded_lap():
    wins = 0
    runs = seeds(500, 3)
    for s in runs:
        e = replay_errors(1000, s)
        wins += np.median(e[100:]) < 20
    assert wins >= (len(runs) + 1) // 2, "localized the car in only {} of {} runs".format(wins, len(runs))
