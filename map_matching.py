""" GPS map matching on the racing line with a hidden Markov model: filtering and the Viterbi algorithm """

import numpy as np


def transition_model(S, step_probs):
    """
    Builds the transition model of the car on the racing line.
    S: number of states (points of the racing line, in driving order, forming a loop)
    step_probs: step_probs[d] = probability that the car moves d points forward in one step
                (d = 0, 1, ..., len(step_probs) - 1); the car never moves backward.
                After point S - 1 comes point 0.

    returns T, a numpy array of shape (S, S) with T[i, j] = P(X_{t+1} = j | X_t = i)

    Example: with S = 5 and step_probs = (0.2, 0.5, 0.3), row 1 is [0, 0.2, 0.5, 0.3, 0] and
    row 4 is [0.5, 0.3, 0, 0, 0.2] (from point 4, one step forward is point 0). Every row sums to 1.
    """
    T = np.zeros((S, S))
    # BEGIN_YOUR_CODE ######################################################
    raise NotImplementedError

    # END_YOUR_CODE ########################################################
    return T


def sensor_likelihood(readings, points, noise_dist, width=None, var=None):
    """
    The sensor model: the likelihood of each GPS reading in each state.
    readings: numpy array (t, 2), GPS readings (x, y)
    points: numpy array (S, 2), the (x, y) position of each state
    noise_dist: the GPS error that the model ASSUMES, independent on the x and y axes:
        "uniform":  uniform on [-width/2, width/2] on each axis:
                    density 1 / width if |error| <= width / 2, and 0 otherwise
        "gaussian": Gaussian with mean 0 and variance var on each axis:
                    density exp(-error^2 / (2 var)) / sqrt(2 pi var)

    returns L, numpy array of shape (t, S), with L[k, j] = p(E_{k+1} = readings[k] | X_{k+1} = j),
    the product of the densities of the x error and the y error, where the error is the
    reading minus the position of the state: (readings[k, 0] - points[j, 0]) on the x axis and
    (readings[k, 1] - points[j, 1]) on the y axis.
    """
    readings, points = np.asarray(readings, float), np.asarray(points, float)
    L = np.zeros((len(readings), len(points)))
    # BEGIN_YOUR_CODE ######################################################
    raise NotImplementedError

    # END_YOUR_CODE ########################################################
    return L


def forward(prior, T, L):
    """
    Filtering (the forward algorithm)
    prior: P(X_0), numpy array of shape (S,)
    T: transition model, T[i, j] = P(X_{t+1} = j | X_t = i), shape (S, S)
    L: sensor likelihoods, L[k, j] = p(e_{k+1} | X_{k+1} = j), shape (t, S)

    returns beliefs, numpy array of shape (t, S), with beliefs[k] = P(X_{k+1} | e_1, ..., e_{k+1})
    Predict with T, update with the likelihood of the reading, and normalize after every step.

    Start from f = prior. For each reading k = 0, ..., t - 1:
      1. predict:    f'(j) = sum_i f(i) T[i, j]        (the car moves one step, also before the first reading)
      2. update:     f''(j) = f'(j) L[k, j]
      3. normalize:  f(j) = f''(j) / sum_j f''(j), and store it in beliefs[k]
    With numpy arrays, the prediction is f @ T.
    """
    t, S = L.shape
    beliefs = np.zeros((t, S))
    # BEGIN_YOUR_CODE ######################################################
    raise NotImplementedError

    # END_YOUR_CODE ########################################################
    return beliefs


def viterbi(prior, T, L):
    """
    The Viterbi algorithm: the most likely sequence of states
    Arguments as in forward.

    returns path, numpy int array of shape (t,), with path[k] = the state at step k+1 in a most likely
    sequence x_1, ..., x_t given e_1, ..., e_t.
    m_k(j) is the probability of the most likely sequence that ends in state j at step k (up to a
    constant factor).
      1. First message: m_1(j) = P(X_1 = j) L[0, j], with P(X_1 = j) = sum_i prior[i] T[i, j].
      2. For every later reading k: m_{k+1}(j) = L[k, j] max_i m_k(i) T[i, j].
         For each j, remember the i that gives the maximum: the back-pointer of state j at this step.
      3. Divide the message by its maximum after every step, so that long sequences do not underflow
         (this does not change which sequence is the most likely one).
      4. The last state of the path is the j with the largest final message. Then go backward: the
         state before it is its back-pointer, and so on to the first step.
    The only difference from forward in step 2 is the maximum over i in place of the sum over i.
    Loops over the states are fine, but numpy is faster: (m[:, None] * T)[i, j] is m[i] T[i, j], and
    np.max(..., axis=0) and np.argmax(..., axis=0) give the maximum over i and its index for every j.
    """
    t, S = L.shape
    path = np.zeros(t, dtype=int)
    # BEGIN_YOUR_CODE ######################################################
    raise NotImplementedError

    # END_YOUR_CODE ########################################################
    return path
