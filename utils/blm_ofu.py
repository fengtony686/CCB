import numpy as np
from utils.blm_lr import find_best_intervention
from utils.blm_lr import find_best_intervention_two_layer


def run_blm_ofu(graph):
    if graph.type == 'parallel':
        matrix_m = np.zeros((graph.num_parents_y, graph.num_parents_y))
        for i in range(graph.num_parents_y):
            matrix_m[i][i] = 1
        matrix_mx = np.zeros(graph.num_parents_y)
        intervened_times = np.zeros(graph.num_parents_y)
        by = np.array([np.zeros(graph.num_parents_y)]).T
        payoff_list = [0]
        total_expected_payoff = 0
        for i in range(graph.T):
            if i >= int(graph.T / 100):
                inverse_m = np.linalg.inv(matrix_m)
                hat_theta_x = np.zeros(graph.num_parents_y)
                for j in range(graph.num_parents_y):
                    hat_theta_x[j] = matrix_mx[j] / (i - intervened_times[j]) if intervened_times[j] < i else 0
                hat_theta_x = np.array([hat_theta_x]).T
                norm_matrix_mx = np.array([np.sqrt(i - intervened_times[j]) for j in range(graph.num_parents_y)])
                hat_theta_y = np.matmul(inverse_m, np.array([by]).T)
                rho = graph.compute_rho_ofu()
                best_intervention = find_best_intervention(graph, inverse_m, hat_theta_x, hat_theta_y, norm_matrix_mx, rho)
            else:
                best_intervention = []
            vy, y = graph.simulate(best_intervention)
            total_expected_payoff += graph.expect_y(best_intervention)
            if i % 100 == 99:
                payoff_list.append(total_expected_payoff)
            matrix_m += np.matmul(vy, vy.T)
            by += y * vy
            for j in best_intervention:
                intervened_times[int(j)] += 1
            for j in range(graph.num_parents_y):
                if j not in best_intervention:
                    matrix_mx[j] += vy[j]
            # print(best_intervention)
        # print(np.matmul(np.linalg.inv(matrix_m), np.array([by]).T))
        return payoff_list, total_expected_payoff
    elif graph.type == 'two layer':
        matrix_m1 = np.zeros((3, 3))
        matrix_m2 = np.zeros((3, 3))
        matrix_m3 = np.zeros((3, 3))
        matrix_my = np.zeros((3, 3))
        for i in range(3):
            matrix_m1[i][i] = 1
            matrix_m2[i][i] = 1
            matrix_m3[i][i] = 1
            matrix_my[i][i] = 1
        matrix_mx = np.zeros(2)
        intervened_times = np.zeros(2)
        b1 = np.array([np.zeros(3)]).T
        b2 = np.array([np.zeros(3)]).T
        b3 = np.array([np.zeros(3)]).T
        by = np.array([np.zeros(3)]).T
        payoff_list = [0]
        total_expected_payoff = 0
        for i in range(graph.T):
            if i >= int(graph.T / 100):
                inverse_m1 = np.linalg.inv(matrix_m1)
                inverse_m2 = np.linalg.inv(matrix_m2)
                inverse_m3 = np.linalg.inv(matrix_m3)
                inverse_my = np.linalg.inv(matrix_my)
                hat_theta_1 = np.zeros(2)
                for j in range(2):
                    hat_theta_1[j] = matrix_mx[j] / (i - intervened_times[j]) if intervened_times[j] < i else 0
                hat_theta_1 = np.array([hat_theta_1]).T
                norm_matrix_mx = np.array([np.sqrt(i - intervened_times[j]) for j in range(2)])
                hat_theta_2 = np.matmul(inverse_m1, np.array([b1]).T)
                hat_theta_3 = np.matmul(inverse_m2, np.array([b2]).T)
                hat_theta_4 = np.matmul(inverse_m3, np.array([b3]).T)
                hat_theta_y = np.matmul(inverse_my, np.array([by]).T)
                rho = graph.compute_rho_ofu()
                best_intervention = find_best_intervention_two_layer(graph, inverse_m1, inverse_m2, inverse_m3, inverse_my,
                                                                     hat_theta_1, hat_theta_2, hat_theta_3, hat_theta_4,
                                                                     hat_theta_y, norm_matrix_mx, rho)
            else:
                best_intervention = []
            vy, y = graph.simulate(best_intervention)
            total_expected_payoff += graph.expect_y(best_intervention)
            if i % 100 == 99:
                payoff_list.append(total_expected_payoff)
            if 0 not in best_intervention:
                matrix_mx[0] += vy[0, 0]
            else:
                intervened_times[0] += 1
            if 1 not in best_intervention:
                matrix_mx[1] += vy[1, 0]
            else:
                intervened_times[1] += 1
            if 2 not in best_intervention:
                matrix_m1 += np.matmul(np.array([[1, vy[0, 0], vy[1, 0]]]).T, np.array([[1, vy[0, 0], vy[1, 0]]]))
                b1 += vy[2, 0] * np.array([[1, vy[0, 0], vy[1, 0]]]).T
            if 3 not in best_intervention:
                matrix_m2 += np.matmul(np.array([[1, vy[0, 0], vy[1, 0]]]).T, np.array([[1, vy[0, 0], vy[1, 0]]]))
                b2 += vy[3, 0] * np.array([[1, vy[0, 0], vy[1, 0]]]).T
            if 4 not in best_intervention:
                matrix_m3 += np.matmul(np.array([[1, vy[0, 0], vy[1, 0]]]).T, np.array([[1, vy[0, 0], vy[1, 0]]]))
                b3 += vy[4, 0] * np.array([[1, vy[0, 0], vy[1, 0]]]).T
            matrix_my += np.matmul(np.array([[vy[2, 0], vy[3, 0], vy[4, 0]]]).T,
                                   np.array([[vy[2, 0], vy[3, 0], vy[4, 0]]]))
            by += y * np.array([[vy[2, 0], vy[3, 0], vy[4, 0]]]).T
        return payoff_list, total_expected_payoff
