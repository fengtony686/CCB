import numpy as np
import math


def run_eps_greedy(graph):
    sum_of_rewards = np.zeros(math.comb(graph.n - 2, graph.k))
    num_of_trigger = np.zeros(math.comb(graph.n - 2, graph.k))
    total_payoff = 0
    payoff_list = [0]
    for i in range(graph.T):
        index = 0
        max_reward = -9999
        best_intervention = []
        best_intervention_index = -1
        random_index = -1
        if np.random.random() > 0.9:
            random_index = int(np.random.random() * len(sum_of_rewards))
        for j in range(np.power(graph.n - 2, graph.k)):
            intervened_indexes = []
            for k in range(graph.k):
                node_index = int(j / np.power(graph.n - 2, k)) % (graph.n - 2)
                if len(intervened_indexes) == 0 or node_index > intervened_indexes[-1]:
                    intervened_indexes.append(node_index)
            if len(intervened_indexes) < graph.k:
                continue
            tmp_reward = sum_of_rewards[index] / num_of_trigger[index] if num_of_trigger[index] > 0 else 1
            if (tmp_reward > max_reward and random_index == -1) or random_index == index:
                max_reward = tmp_reward
                best_intervention = intervened_indexes
                best_intervention_index = index
            index += 1
        _, y = graph.simulate(best_intervention)
        sum_of_rewards[best_intervention_index] += y
        num_of_trigger[best_intervention_index] += 1
        total_payoff += graph.expect_y(best_intervention)
        if i % 100 == 99:
            payoff_list.append(total_payoff)
    # print(sum_of_rewards / num_of_trigger)
    return payoff_list, total_payoff


# if __name__ == "__main__":
#     from parallel_graph import ParallelGraph
#
#     newGraph = ParallelGraph([.3, .4, .2, .1, .6, .4], [.1, .3, .2, .2, .1, .1], 20000, 3)
#     _, payoff = run_eps_greedy(newGraph)
#     print(newGraph.best_expect_y() * 20000 - payoff)

if __name__ == "__main__":
    from two_layer_graph import TwoLayerGraph

    newGraph = TwoLayerGraph(20000)
    _, payoff = run_eps_greedy(newGraph)
    print(newGraph.best_expect_y() * 20000 - payoff)
