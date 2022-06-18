# Combinatorial Causal Bandits
This repository contains code of numerical experiments for paper:         
**Combinatorial Causal Bandits**        
[Shi Feng](https://fengshi.link/), [Wei Chen](https://www.microsoft.com/en-us/research/people/weic/)          
[[ArXiv Version](https://arxiv.org/abs/2206.01995)]

## Usage
$G_1,G_2,\cdots,G_4$ are parallel binary linear models. $G_1$ and $G_2$ are shown as below:
<center>
    <img src="https://github.com/fengtony686/CCB/raw/main/results/G1_structure.png" width="300"/><img src="https://github.com/fengtony686/CCB/raw/main/results/G2_structure.png" width="300"/>
</center>

$G_3$ is $G_2$ without $X_8$ and $X_9$. $G_4$ is $G_2$ removing $X_6,X_7,X_8$ and $X_9$.
$G_5$ is a two-layer BLM shown as below:
![G5](https://github.com/fengtony686/CCB/raw/main/results/G5_structure.png)

If you want to compare regrets of BLM-OFU, BLM-LR, UCB and $\epsilon$-greedy algorithms on graph $G_\*$, you need to run
```
python main.py --G*
```
You can find our running samples in `./results/` directory.

## File Hierarchy

```
.
├── utils/                    # implementations of BLM and online algorithms
│   ├── blm_lr.py             # implementing BLM-LR algorithm
│   ├── blm_ofu.py            # implementing BLM-OFU (BGLM-OFU) algorithm
│   ├── epsilon_greedy.py     # implementing epsilon-greedy algorithm
│   ├── ucb.py                # implementing UCB algorithm
│   ├── parallel_graph.py     # implementing parallel graph (G1, G2, G3, G4)
│   └── two_layer_graph.py    # implementing two-layer graph (G5)
├── results/                  # our running samples
├── main.py                   # main file
├── .gitignore                # exclude some annoying files from git
├── LICENSE                   # MIT license
└── README.md                 # what you are reading now
```

## Contact

If you have any questions, feel free to contact us through email (fengs19@mails.tsinghua.edu.cn) or Github issues. Enjoy!
