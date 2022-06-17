import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from utils.parallel_graph import ParallelGraph
from utils.two_layer_graph import TwoLayerGraph
from utils.ucb import run_ucb
from utils.epsilon_greedy import run_eps_greedy
from utils.blm_lr import run_blm_lr
from utils.blm_ofu import run_blm_ofu
import seaborn as sns
from multiprocessing import Process, Queue


def compute_regret_list(repeat_num, best_y, graph, algorithm, q):
    regret_list = np.empty((repeat_num, int(graph.T / 100 + 1)))
    for i in trange(repeat_num):
        for j in trange(30):
            if algorithm == "blm-lr":
                payoff_list, _ = run_blm_lr(graph)
            elif algorithm == 'blm-ofu':
                payoff_list, _ = run_blm_ofu(graph)
            elif algorithm == 'ucb':
                payoff_list, _ = run_ucb(graph)
            else:
                payoff_list, _ = run_eps_greedy(graph)
            for index, k in enumerate(payoff_list):
                payoff_list[index] = best_y * 100 * index - payoff_list[index]
            regret_list[i, ::] = payoff_list
    q.put(regret_list)


def compute_confidence_region(regret_list):
    average_regret = np.average(regret_list, axis=0)
    std_regret = np.std(regret_list, axis=0)
    repeated_times = regret_list.shape[0]
    return average_regret, average_regret + 1.96 * std_regret / repeated_times, average_regret - 1.96 * std_regret / repeated_times


if __name__ == "__main__":
    lr_queue = Queue()
    ofu_queue = Queue()
    ucb_queue = Queue()
    eps_greedy_queue = Queue()
    repeated_num = 20
    T = 10000

    # newGraph = ParallelGraph([.2, .2, .6, .6, .6, .6, .6, .6], [.2, .2, .1, .1, .1, .1, .1, .1], T, 2)
    # newGraph = ParallelGraph([.3, .4, .2, .1, .6, .5], [.1, .3, .2, .2, .1, .1], T, 3)
    # newGraph = ParallelGraph([.2, .2, .6, .6, .6, .6], [.2, .2, .1, .1, .1, .1], T, 2)
    # newGraph = ParallelGraph([.2, .2, .6, .6], [.2, .2, .1, .1], T, 2)
    newGraph = TwoLayerGraph(T)

    best_expect_y = newGraph.best_expect_y()

    thread1 = Process(target=compute_regret_list,
                      args=(repeated_num, best_expect_y, newGraph, "blm-lr", lr_queue))
    thread2 = Process(target=compute_regret_list,
                      args=(repeated_num, best_expect_y, newGraph, "blm-ofu", ofu_queue))
    thread3 = Process(target=compute_regret_list,
                      args=(repeated_num, best_expect_y, newGraph, "ucb", ucb_queue))
    thread4 = Process(target=compute_regret_list,
                      args=(repeated_num, best_expect_y, newGraph, "eps-greedy", eps_greedy_queue))
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    lr_regret_list = lr_queue.get()
    ofu_regret_list = ofu_queue.get()
    ucb_regret_list = ucb_queue.get()
    eps_greedy_regret_list = eps_greedy_queue.get()

    # lr_regret_list = compute_regret_list(repeated_num, best_expect_y, newGraph, "blm-lr")
    # ofu_regret_list = compute_regret_list(repeated_num, best_expect_y, newGraph, "blm-ofu")
    # ucb_regret_list = compute_regret_list(repeated_num, best_expect_y, newGraph, "ucb")
    # eps_greedy_regret_list = compute_regret_list(repeated_num, best_expect_y, newGraph, "eps-greedy")

    clrs = sns.color_palette("husl", 5)
    plt.rcParams['figure.figsize'] = (20.0, 15.0)

    avg_lr_regret_list, max_lr_regret_list, min_lr_regret_list = compute_confidence_region(lr_regret_list)
    tmp = [avg_lr_regret_list[int((len(avg_lr_regret_list) / 20)) * (i + 1)] for i in range(20)]
    tmp = [0] + tmp
    plt.scatter(np.arange(21) * (T / 20), tmp, marker='+', label="BLM-LR", c=clrs[0])
    plt.plot(np.arange(int(T / 100 + 1)) * 100, np.array(avg_lr_regret_list), c=clrs[0])
    plt.fill_between(np.arange(int(T / 100 + 1)) * 100, min_lr_regret_list, max_lr_regret_list, alpha=0.3,
                     facecolor=clrs[0])

    avg_ofu_regret_list, max_ofu_regret_list, min_ofu_regret_list = compute_confidence_region(ofu_regret_list)
    tmp = [avg_ofu_regret_list[int((len(avg_ofu_regret_list) / 20)) * (i + 1)] for i in range(20)]
    tmp = [0] + tmp
    plt.scatter(np.arange(21) * (T / 20), tmp, marker='x', label="BLM-OFU", c=clrs[1])
    plt.plot(np.arange(int(T / 100 + 1)) * 100, np.array(avg_ofu_regret_list), c=clrs[1])
    plt.fill_between(np.arange(int(T / 100 + 1)) * 100, min_ofu_regret_list, max_ofu_regret_list, alpha=0.3,
                     facecolor=clrs[1])

    avg_ucb_regret_list, max_ucb_regret_list, min_ucb_regret_list = compute_confidence_region(ucb_regret_list)
    tmp = [avg_ucb_regret_list[int((len(avg_ucb_regret_list) / 20)) * (i + 1)] for i in range(20)]
    tmp = [0] + tmp
    plt.scatter(np.arange(21) * (T / 20), tmp, marker='o', label="UCB", c=clrs[2])
    plt.plot(np.arange(int(T / 100 + 1)) * 100, np.array(avg_ucb_regret_list), c=clrs[2])
    plt.fill_between(np.arange(int(T / 100 + 1)) * 100, min_ucb_regret_list, max_ucb_regret_list, alpha=0.3,
                     facecolor=clrs[2])

    avg_eps_greedy_regret_list, max_eps_greedy_regret_list, min_eps_greedy_regret_list = compute_confidence_region(
        eps_greedy_regret_list)
    tmp = [avg_eps_greedy_regret_list[int((len(avg_eps_greedy_regret_list) / 20)) * (i + 1)] for i in range(20)]
    tmp = [0] + tmp
    plt.scatter(np.arange(21) * (T / 20), tmp, marker='D', label=r"$\epsilon$-greedy", c=clrs[3])
    plt.plot(np.arange(int(T / 100 + 1)) * 100, np.array(avg_eps_greedy_regret_list),
             c=clrs[3])
    plt.fill_between(np.arange(int(T / 100 + 1)) * 100, min_eps_greedy_regret_list, max_eps_greedy_regret_list,
                     alpha=0.3, facecolor=clrs[3])

    plt.xlabel("Round Number")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.show()
