"""
Generates plots for particle filtering and Kalman filtering using a pre-recorded drive around the track,
and for map matching with filtering and the Viterbi algorithm (Part 4) on simulated drives.

    python plots.py -w pf --num_particles 100 --max_sensor_range 100 --sensor_noise_std 2.0 --filename pf.png
    python plots.py -w kf --gps_noise_dist gaussian --gps_noise_var 4.0 --filename kf.png
    python plots.py -w hmm --gps_noise_dist uniform --gps_noise_width 60 --filename hmm.png

Plots are stored inside the plots directory.  Run python plots.py -h for all options.
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib
import numpy as np

from simulator import Simulator, WORLD_WIDTH, WORLD_HEIGHT, CAR2_DELAY

FONT = {"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 12}  # larger fonts so the plots are readable in a report
FIRST_TURN_EXIT = 62     # replay step at which car 1 leaves the first turn (car 2: CAR2_DELAY / 2 steps later)
LOCALIZED_RADIUS = 20.0  # a filter "has" the car while its position error is below this


def localized(errors, car):
    """True if the error stays below LOCALIZED_RADIUS from the end of the first turn to the end of the lap."""
    start = FIRST_TURN_EXIT + (CAR2_DELAY // 2 if car == 2 else 0)
    return bool(np.all(np.asarray(errors[start:]) < LOCALIZED_RADIUS))


def run(args, seed):
    """Replay the recorded lap once with the given settings; returns the simulator and the time per step."""
    np.random.seed(seed)
    random.seed(seed)
    if args.which == "pf":
        sim = Simulator(max_sensor_range=args.max_sensor_range, sensor_std=args.sensor_noise_std,
                        num_particles=args.num_particles, sensor_noise_dist=args.sensor_noise_dist,
                        sensor_noise_bound=args.sensor_noise_bound, sensor_fail_prob=args.sensor_fail_prob,
                        assumed_noise_dist=args.assumed_noise_dist, assumed_noise_std=args.assumed_noise_std,
                        assumed_noise_bound=args.assumed_noise_bound, assumed_fail_prob=args.assumed_fail_prob)
        sim.toggle_particles()
    else:
        sim = Simulator(gps_noise_var=args.gps_noise_var, gps_noise_width=args.gps_noise_width,
                        assumed_gps_var=args.assumed_gps_var, assumed_gps_width=args.assumed_gps_width)
        sim.gps_noise_dist = args.gps_noise_dist
        sim.toggle_kalman()
    sim.toggle_replay()
    truth = {1: [], 2: []}
    est = {1: [], 2: []}
    gps = {1: [], 2: []}
    sd = {1: [], 2: []}
    t0 = time.perf_counter()
    steps = 0
    while sim.loop():
        steps += 1
        truth[1].append(sim.car1.pos.copy())
        truth[2].append(sim.car2.pos.copy())
        if args.which == "pf":
            est[1].append([sim.x_est1, sim.y_est1])
            est[2].append([sim.x_est2, sim.y_est2])
        else:
            est[1].append(np.ravel(sim.kf_state1)[:2])
            est[2].append(np.ravel(sim.kf_state2)[:2])
            gps[1].append(sim.gps_measurement1.copy())
            gps[2].append(sim.gps_measurement2.copy())
            for k, f in ((1, sim.kalman_filter1), (2, sim.kalman_filter2)):
                P = np.asarray(f.kf.P, dtype=float)
                sd[k].append(np.sqrt(P[0, 0] + P[1, 1]))
    ms_per_step = 1000 * (time.perf_counter() - t0) / max(steps, 1)
    data = {k: dict(truth=np.array(truth[k]), est=np.array(est[k]), gps=np.array(gps[k]), sd=np.array(sd[k]),
                    errors=np.array(sim.car1_errors if k == 1 else sim.car2_errors)) for k in (1, 2)}
    return data, ms_per_step


def summary(args, data, ms_per_step, seed):
    if args.which == "pf":
        parts = []
        for k in (1, 2):
            e = data[k]["errors"]
            start = FIRST_TURN_EXIT + (CAR2_DELAY // 2 if k == 2 else 0)
            ok = localized(e, k)
            parts.append("car {}: {} (mean error after the first turn {:.2f})".format(
                k, "localized" if ok else "NOT localized", float(np.mean(e[start:]))))
        return "seed {}: ".format(seed) + ", ".join(parts) + ", {:.1f} ms per step".format(ms_per_step)
    parts = []
    for k in (1, 2):
        d = data[k]
        gps_err = np.linalg.norm(d["gps"] - d["truth"], axis=1)
        parts.append("car {}: RMS error GPS {:.2f}, filter {:.2f} (the filter expects {:.2f})".format(
            k, float(np.sqrt(np.mean(gps_err ** 2))), float(np.sqrt(np.mean(d["errors"] ** 2))),
            float(np.sqrt(np.mean(d["sd"] ** 2)))))
    return "seed {}: ".format(seed) + ", ".join(parts)


def plot(args, data):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    bg = plt.imread("data/track.png")
    for col, k in enumerate((1, 2)):
        d = data[k]
        ax = axs[0, col]
        ax.imshow(bg, extent=[0, WORLD_WIDTH, WORLD_HEIGHT, 0], alpha=0.5)
        ax.plot(d["truth"][:, 0], d["truth"][:, 1], color="black", label="car position")
        if args.which == "kf":
            ax.plot(d["gps"][:, 0], d["gps"][:, 1], ".", color="red", ms=2, label="GPS measurement")
        ax.plot(d["est"][:, 0], d["est"][:, 1], color="blue", lw=1, label="estimated position")
        ax.set_xlim(0, WORLD_WIDTH)
        ax.set_ylim(WORLD_HEIGHT, 0)
        ax.set_title("Path - Car {}".format(k))
        ax.legend(loc="lower left")
        ax = axs[1, col]
        ax.plot(d["errors"])
        ax.set_xlabel("Time")
        ax.set_ylabel("Error (distance between estimate and car)")
        ax.set_title("Error - Car {}".format(k))
        if args.which == "kf":
            ax.plot(d["sd"], "--", lw=1, label="error the filter expects, sqrt(var x + var y)")
            ax.legend()
        if args.which == "pf":
            ax.set_yscale("log")
            ax.axhline(LOCALIZED_RADIUS, color="gray", ls="--", lw=0.8)
            ax.axvline(FIRST_TURN_EXIT + (CAR2_DELAY // 2 if k == 2 else 0), color="gray", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/" + args.filename, bbox_inches="tight")


def hmm_run(args, seed):
    """One drive on the racing line: filtering and the Viterbi path from the GPS readings."""
    import track_hmm as H
    from map_matching import transition_model, sensor_likelihood, forward, viterbi
    points = H.racing_line()
    S = len(points)
    d = H.simulate_drive(seed, points, args.gps_noise_dist, args.gps_noise_width, args.gps_noise_var)
    adist = args.assumed_gps_dist or args.gps_noise_dist
    awidth = args.assumed_gps_width if args.assumed_gps_width is not None else args.gps_noise_width
    avar = args.assumed_gps_var if args.assumed_gps_var is not None else args.gps_noise_var
    prior, T = H.uniform_prior(S), transition_model(S, H.STEP_PROBS)
    L = sensor_likelihood(d["readings"], points, adist, awidth, avar)
    res = dict(drive=d, points=points,
               assumed="uniform, width {:g}".format(awidth) if adist == "uniform" else "gaussian, var {:g}".format(avar),
               gps_error=np.linalg.norm(d["readings"] - d["true_xy"], axis=1))
    res["impossible_at"] = H.first_impossible_step(prior, T, L)
    if res["impossible_at"] is not None:
        return res
    filt = forward(prior, T, L).argmax(axis=1)
    path = viterbi(prior, T, L)
    res.update(filtering=filt, viterbi=path,
               filtering_error=H.position_error(filt, points, d["true_xy"]),
               viterbi_error=H.position_error(path, points, d["true_xy"]),
               filtering_impossible=H.impossible_moves(filt, S), viterbi_impossible=H.impossible_moves(path, S))
    return res


def hmm_summary(seed, res):
    if res["impossible_at"] is not None:
        return ("drive {}: the readings are impossible under the assumed GPS model: at step {} every state "
                "sequence has probability 0".format(seed, res["impossible_at"] + 1))
    return ("drive {}: mean error  GPS {:.2f}  filtering {:.2f}  Viterbi {:.2f};  impossible moves  filtering {}  "
            "Viterbi {}".format(seed, res["gps_error"].mean(), res["filtering_error"].mean(),
                                res["viterbi_error"].mean(), res["filtering_impossible"], res["viterbi_impossible"]))


def hmm_plot(args, res):
    import matplotlib.pyplot as plt
    import track_hmm as H
    d, P = res["drive"], res["points"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax1.imshow(plt.imread("data/track.png"), extent=[0, WORLD_WIDTH, WORLD_HEIGHT, 0], alpha=0.4)
    ax1.plot(P[:, 0], P[:, 1], ".", color="gray", ms=2, label="states (racing line)")
    ax1.plot(d["readings"][:, 0], d["readings"][:, 1], "x", color="tab:green", ms=4, label="GPS reading")
    ax1.plot(d["true_xy"][:, 0], d["true_xy"][:, 1], "o", color="black", ms=3, label="true position")
    t = np.arange(1, len(d["states"]) + 1)
    if res["impossible_at"] is None:
        f = P[res["filtering"]]
        v = P[res["viterbi"]]
        ax1.plot(f[:, 0], f[:, 1], "-", color="tab:blue", lw=1, label="most likely state, filtering")
        ax1.plot(v[:, 0], v[:, 1], "--", color="tab:red", lw=1.5, label="Viterbi path")
        S = len(P)

        def offset(path):  # signed distance along the racing line from the true state, in points
            return (np.asarray(path) - d["states"] + S // 2) % S - S // 2
        ax2.axhline(0, color="black", lw=1, label="true state")
        ax2.step(t, offset(res["filtering"]), where="mid", color="tab:blue", lw=1, label="most likely state, filtering")
        ax2.step(t, offset(res["viterbi"]), "--", where="mid", color="tab:red", lw=1.5, label="Viterbi path")
        bad = np.where(np.mod(np.diff(res["filtering"]), S) > 2)[0] + 1
        ax2.plot(t[bad], offset(res["filtering"])[bad], "v", color="tab:blue", ms=6,
                 label="impossible move, filtering")
    else:
        ax2.text(0.5, 0.5, "impossible readings at step {}".format(res["impossible_at"] + 1),
                 ha="center", transform=ax2.transAxes)
    ax1.set_xlim(0, WORLD_WIDTH)
    ax1.set_ylim(WORLD_HEIGHT, 0)
    ax1.legend(loc="lower left")
    real = ("uniform, width {:g}".format(args.gps_noise_width) if args.gps_noise_dist == "uniform"
            else "gaussian, var {:g}".format(args.gps_noise_var))
    ax1.set_title("Drive {}: GPS noise {}; the model assumes {}".format(args.seed, real, res["assumed"]))
    ax2.set_xlabel("step")
    ax2.set_ylabel("estimate - true state (points, {:g} px apart)".format(H.SPACING))
    ax2.set_title("Estimate relative to the true state (positive: ahead)")
    if res["impossible_at"] is None:
        ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/" + args.filename, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--which", required=True, choices=("pf", "kf", "hmm"),
                        help="which filter to run (pf: particle filter, kf: Kalman filter, hmm: map matching)")
    parser.add_argument("-n", "--num_particles", default=50, type=int, help="Number of particles for particle filtering")
    parser.add_argument("-m", "--max_sensor_range", default=50, type=int, help="Maximum range of the car's sensors")
    parser.add_argument("-s", "--sensor_noise_std", default=0.0, type=float, help="Std dev of car's sensor noise")
    parser.add_argument("--sensor_noise_dist", default="gaussian", choices=("gaussian", "uniform"),
                        help="Distribution of the car's sensor noise")
    parser.add_argument("--sensor_noise_bound", default=5.0, type=float,
                        help="Bound b of the car's uniform sensor noise on [-b, b]")
    parser.add_argument("--sensor_fail_prob", default=0.0, type=float,
                        help="Probability that a reading of the car's sensors is a random value")
    parser.add_argument("--assumed_noise_dist", default="gaussian", choices=("gaussian", "uniform"),
                        help="Sensor noise distribution that the particle filter assumes")
    parser.add_argument("--assumed_noise_std", default=10.0, type=float,
                        help="Std dev of the sensor noise that the particle filter assumes")
    parser.add_argument("--assumed_noise_bound", default=5.0, type=float,
                        help="Bound of the uniform sensor noise that the particle filter assumes")
    parser.add_argument("--assumed_fail_prob", default=0.0, type=float,
                        help="Probability of a random reading that the particle filter assumes")
    parser.add_argument("-d", "--gps_noise_dist", default="gaussian", choices=("gaussian", "uniform"),
                        help="Type of distribution for GPS sensor noise (gaussian or uniform)")
    parser.add_argument("-gv", "--gps_noise_var", default=10.0, type=float,
                        help="Variance of gaussian noise for GPS measurement (Kalman filter, map matching)")
    parser.add_argument("-gw", "--gps_noise_width", default=20, type=float,
                        help="Width of uniformly random noise for GPS measurement (Kalman filter, map matching)")
    parser.add_argument("--assumed_gps_var", default=None, type=float,
                        help="GPS noise variance that the Kalman filter / map matching assumes (default: the real one)")
    parser.add_argument("--assumed_gps_width", default=None, type=float,
                        help="GPS noise width that the Kalman filter / map matching assumes (default: the real one)")
    parser.add_argument("--assumed_gps_dist", default=None, choices=("gaussian", "uniform"),
                        help="GPS noise distribution that the map-matching model assumes (default: the real one)")
    parser.add_argument("--seed", default=0, type=int, help="Random seed of the run (the drive number for hmm)")
    parser.add_argument("--trials", default=1, type=int,
                        help="Run seeds seed, seed+1, ... (this many) and report each; the plot shows the first")
    parser.add_argument("-f", "--filename", default="plot.png", help="name of image to store plot inside the plots/ directory")
    args = parser.parse_args()
    matplotlib.use("Agg")
    matplotlib.rcParams.update(FONT)

    if args.which == "hmm":
        results = []
        for t in range(args.trials):
            res = hmm_run(args, args.seed + t)
            results.append(res)
            print(hmm_summary(args.seed + t, res))
            if t == 0:
                hmm_plot(args, res)
        if args.trials > 1:
            ok = [r for r in results if r["impossible_at"] is None]
            if len(ok) < len(results):
                print("{} of {} drives: the readings are impossible under the assumed GPS model (no filtering or "
                      "Viterbi result)".format(len(results) - len(ok), len(results)))
            if ok:
                print("average over {} drives: mean error  GPS {:.2f}  filtering {:.2f}  Viterbi {:.2f};  impossible "
                      "moves  filtering {:.2f}  Viterbi {:.2f}".format(
                          len(ok), np.mean([r["gps_error"].mean() for r in ok]),
                          np.mean([r["filtering_error"].mean() for r in ok]),
                          np.mean([r["viterbi_error"].mean() for r in ok]),
                          np.mean([r["filtering_impossible"] for r in ok]),
                          np.mean([r["viterbi_impossible"] for r in ok])))
        print("plot saved in plots/" + args.filename)
        return
    count = {1: 0, 2: 0}
    for t in range(args.trials):
        seed = args.seed + t
        data, ms = run(args, seed)
        print(summary(args, data, ms, seed))
        if args.which == "pf":
            for k in (1, 2):
                count[k] += localized(data[k]["errors"], k)
        if t == 0:
            plot(args, data)
    if args.which == "pf" and args.trials > 1:
        print("localized: car 1 in {} of {} runs, car 2 in {} of {} runs".format(count[1], args.trials, count[2], args.trials))
    print("plot saved in plots/" + args.filename)


if __name__ == "__main__":
    main()
