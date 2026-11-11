""" Bayesian networks """

import csv

from probability import BayesNet, elimination_ask

T, F = True, False


class DataPoint:
    """
    Represents a single datapoint gathered from one overtake attempt.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win


def load_data(fname="data/bn_data.csv"):
    """
    Loads the dataset of overtake attempts
    returns a list of DataPoint objects
    """
    data = []
    with open(fname) as f:
        for row in csv.DictReader(f):
            v = {k: row[k] == "T" for k in row}
            data.append(DataPoint(v["MuchFaster"], v["Early"], v["Overtake"], v["Crash"], v["Win"]))
    return data


def generate_bayesnet(data=None):
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    data: a list of DataPoint objects (by default, the dataset in data/bn_data.csv)
    returns the BayesNet object

    Each DataPoint d is one overtake attempt, with boolean attributes d.muchfaster, d.early,
    d.overtake, d.crash, and d.win. Estimate every probability as a frequency in data, e.g.
        P(Crash = T | MuchFaster = T, Early = F)
            = (number of d with d.crash, d.muchfaster and not d.early)
              / (number of d with d.muchfaster and not d.early)

    Add one node per variable to bayes_net, parents before children, with
        bayes_net.add((name, parents, cpt))
    name:    the variable name: "MuchFaster", "Early", "Overtake", "Crash", or "Win"
    parents: the names of the parents in one string, separated by spaces ("" if there are none)
    cpt:     P(variable = True): a number if there are no parents; otherwise a dictionary from the
             values of the parents, in the order of the parents string, to the probability, e.g.
             {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001} for two parents,
             or {T: 0.90, F: 0.05} for one parent
    (Equivalently, build the network at once with BayesNet([node, node, ...]).)
    """
    bayes_net = BayesNet()
    # load the dataset, a list of DataPoint objects
    if data is None:
        data = load_data()
    # BEGIN_YOUR_CODE ######################################################
    raise NotImplementedError

    # END_YOUR_CODE ########################################################
    return bayes_net


def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 2:
    the values of (MuchFaster, Early) with the highest probability of winning the race without crashing
    Returns the optimal values as a tuple of booleans (MuchFaster, Early)

    Compute the probability for each of the four conditions and return the best one.
    To query the network, use elimination_ask (variable elimination, from probability.py):
        dist = elimination_ask("Win", dict(MuchFaster=T, Early=F), bayes_net)
    returns the distribution P(Win | MuchFaster = T, Early = F); dist[T] is the probability that
    Win is True and dist[F] the probability that it is False. The evidence dictionary may contain
    any of the other variables.
    """
    # BEGIN_YOUR_CODE ######################################################
    raise NotImplementedError

    # END_YOUR_CODE ########################################################


def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0], cond[1]))


if __name__ == "__main__":
    main()
