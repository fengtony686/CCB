import numpy as np


class TwoLayerGraph:
    def __init__(self, num_of_rounds, k=2):
        self.type = 'two layer'
        self.prob1 = .1 * np.ones(5)
        self.prob2 = np.array([.1, .2])
        self.prob3 = np.array([.7, .1])
        self.prob4 = np.array([.7, .1])
        self.proby = np.array([.6, .1, .1])
        self.n = 7
        self.delta = 1 / self.n / np.sqrt(num_of_rounds)
        self.T = num_of_rounds
        self.k = k

    def compute_rho_lr(self, t):
        return (np.sqrt(self.n * np.log(1 + self.n * t) + 2 * np.log(1 / self.delta)) + np.sqrt(self.n)) / 10

    def compute_rho_ofu(self):
        return 3 * np.sqrt(np.log(3 / self.delta)) / 10

    def best_expect_y(self):
        return self.expect_y([0, 2])

    def simulate(self, intervention):
        x1 = float(np.random.random() < self.prob1[0]) if 0 not in intervention else 1
        x2 = float(np.random.random() < self.prob1[1]) if 1 not in intervention else 1
        x3 = float(np.random.random() < self.prob1[2] + x1 * self.prob2[0] + x2 * self.prob2[
            1]) if 2 not in intervention else 1
        x4 = float(np.random.random() < self.prob1[3] + x1 * self.prob3[0] + x2 * self.prob3[
            1]) if 3 not in intervention else 1
        x5 = float(np.random.random() < self.prob1[4] + x1 * self.prob4[0] + x2 * self.prob4[
            1]) if 4 not in intervention else 1
        y = float(np.random.random() < x3 * self.proby[0] + x4 * self.proby[1] + x5 * self.proby[2])
        return np.array([[x1, x2, x3, x4, x5]]).T, y

    def expect_y(self, intervention):
        x1 = self.prob1[0] if 0 not in intervention else 1
        x2 = self.prob1[1] if 1 not in intervention else 1
        x3 = self.prob1[2] + x1 * self.prob2[0] + x2 * self.prob2[1] if 2 not in intervention else 1
        x4 = self.prob1[3] + x1 * self.prob3[0] + x2 * self.prob3[1] if 3 not in intervention else 1
        x5 = self.prob1[4] + x1 * self.prob4[0] + x2 * self.prob4[1] if 4 not in intervention else 1
        y = x3 * self.proby[0] + x4 * self.proby[1] + x5 * self.proby[2]
        return y
