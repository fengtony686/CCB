import numpy as np


class ParallelGraph:
    def __init__(self, theta_x, theta_y, num_of_rounds, k):
        self.type = 'parallel'
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.n = len(theta_x) + 2
        self.num_parents_y = self.n - 2
        self.delta = 1 / self.n / np.sqrt(num_of_rounds)
        self.T = num_of_rounds
        self.k = k

    def compute_rho_lr(self, t):
        return (np.sqrt(self.n * np.log(1 + self.n * t) + 2 * np.log(1 / self.delta)) + np.sqrt(self.n)) / 10

    def compute_rho_ofu(self):
        return 3 * np.sqrt(np.log(3 / self.delta)) / 10

    def simulate(self, intervention):
        x = [np.random.random() < self.theta_x[i] for i in range(self.num_parents_y)]
        for i in intervention:
            x[int(i)] = 1
        y = (np.random.random() < np.sum(np.array(x) * self.theta_y))
        return np.array([x]).T, y

    def expect_y(self, intervention):
        x = self.theta_x.copy()
        for i in intervention:
            x[int(i)] = 1
        return np.sum(np.array(x) * self.theta_y)

    def best_expect_y(self):
        max_y = - 9999
        for j in range(np.power(self.num_parents_y, self.k)):
            intervened_indexes = []
            for k in range(self.k):
                index = int(j / np.power(self.num_parents_y, k)) % self.num_parents_y
                if index not in intervened_indexes:
                    intervened_indexes.append(index)
            if len(intervened_indexes) < self.k:
                continue
            expect_y = self.expect_y(intervened_indexes)
            if expect_y > max_y:
                max_y = expect_y
        return max_y
