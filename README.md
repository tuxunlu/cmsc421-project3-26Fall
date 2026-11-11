# CMSC 421 Project 3: Dynamic Bayes Nets and Kalman Filters

The handout (PDF) has the full assignment. This file is a quick reference.

## Setup

Use **Python 3.12**, in a virtual environment created with that interpreter (Python 3.10 and 3.11
also work, but 3.12 is the version the instructions below install). Follow the steps for your
system; they clone the repository, install the requirements, open the GUI, and run the public tests.

In every later terminal, activate the environment before running the code:

```bash
cd cmsc421-project3-26Fall
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

`which python` should then end in `cmsc421-project3-26Fall/.venv/bin/python`.

### macOS, Apple Silicon (M1, M2, M3, M4)

```bash
/opt/homebrew/bin/brew install python@3.12 python-tk@3.12
git clone https://github.com/tuxunlu/cmsc421-project3-26Fall.git
cd cmsc421-project3-26Fall
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -c "import ssl, tkinter; print(ssl.OPENSSL_VERSION, '| Tk', tkinter.TkVersion)"
python -m pip install -r requirements.txt
python gui.py                      # drive the cars with the keyboard
python -m pytest                   # public tests: PASS / FAIL / INCOMPLETE per part
```

- The `import` line should print an OpenSSL version and Tk 8.6 or newer.
- Call Homebrew as `/opt/homebrew/bin/brew`. If `which brew` prints `/usr/local/bin/brew`, that is a
  second, Intel Homebrew, and it does not install the interpreter used above. If
  `/opt/homebrew/bin/brew` is not found, install Homebrew from https://brew.sh and rerun the first command.
- Create `.venv` with `/opt/homebrew/bin/python3.12`, not with `python3`. `/usr/bin/python3` is the
  interpreter of the Xcode Command Line Tools: its Tk is too old for the GUI (the window stays
  empty), and packages installed for it can be built for the wrong architecture.

### macOS, Intel

Install Python 3.12 from https://www.python.org/downloads/macos/ (the installer includes Tk), then:

```bash
git clone https://github.com/tuxunlu/cmsc421-project3-26Fall.git
cd cmsc421-project3-26Fall
python3.12 -m venv .venv
source .venv/bin/activate
python -c "import ssl, tkinter; print(ssl.OPENSSL_VERSION, '| Tk', tkinter.TkVersion)"
python -m pip install -r requirements.txt
python gui.py
python -m pytest
```

### Linux (Debian or Ubuntu)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-tk
git clone https://github.com/tuxunlu/cmsc421-project3-26Fall.git
cd cmsc421-project3-26Fall
python3.12 -m venv .venv
source .venv/bin/activate
python -c "import ssl, tkinter; print(ssl.OPENSSL_VERSION, '| Tk', tkinter.TkVersion)"
python -m pip install -r requirements.txt
python gui.py
python -m pytest
```

If your release does not have a `python3.12` package, use its `python3` (3.10 or newer) with
`python3-venv` and `python3-tk`.

### Windows

Install Python 3.12 from https://www.python.org/downloads/windows/ with "tcl/tk" and "Add python.exe
to PATH" selected. In Command Prompt, from the folder where you want the project:

```
git clone https://github.com/tuxunlu/cmsc421-project3-26Fall.git
cd cmsc421-project3-26Fall
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python gui.py
python -m pytest
```

## Files you will edit

| File | Part |
| --- | --- |
| `particle_filter.py` | Part 1: particle filter |
| `bayesian_network.py` | Part 2: Bayesian network |
| `kalman_filter.py` | Part 3: Kalman filter |
| `map_matching.py` | Part 4: map matching with a hidden Markov model (filtering, Viterbi) |

Write your code between `# BEGIN_YOUR_CODE` and `# END_YOUR_CODE`. You may also edit
`plots.py` or add your own plotting scripts.

## Files you will not edit

| File | Contents |
| --- | --- |
| `gui.py` | GUI for interacting with the simulator |
| `simulator.py`, `car.py`, `racetrack.py`, `utils.py` | Simulator, cars, racetrack, utility functions |
| `track_hmm.py` | Part 4: the racing-line states and simulated drives with GPS readings |
| `probability.py` | Code from the AIMA repository (Bayesian networks) |
| `data/` | Racetrack, recorded lap, overtake dataset, images |
| `tests/` | Public tests |

## Plots

```bash
python plots.py -w pf --num_particles 100 --max_sensor_range 100 --sensor_noise_std 2.0 --filename pf.png
python plots.py -w kf --gps_noise_dist gaussian --gps_noise_var 4.0 --filename kf.png
python plots.py -w hmm --gps_noise_dist uniform --gps_noise_width 60 --filename hmm.png
python plots.py -h            # all options
```
