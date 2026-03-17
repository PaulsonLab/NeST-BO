# [AISTATS 2026] NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information

This repository contains the code to reproduce the NeST-BO and NeST-BO-sub algorithms proposed in the paper _NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information_. 

NeST-BO has been published as a conference paper in AISTATS 2026.

# Installation
```sh
pip install -r requirements.txt
```

## Running Experiments

Experiments can be run using the `main_NeSTBO.py` and `main_NeSTBO_sub.py` script. You must specify a benchmark to run the algorithms.

**Basic Command**
```
python main_NeSTBO.py benchmark=<benchmark_name>
python main_NeSTBO_sub.py benchmark=<benchmark_name>
```

*   To see a list of available benchmarks, run `python main_NeSTBO.py`.
*   Adding `seed=<number>` is recommended for reproducibility.

**Configuration Overrides**
All default settings are stored in configs/default.yaml. Since this project uses [Hydra](https://hydra.cc/), you have the flexibility to modify these values on the fly via the command line without editing the file.

```
# Example: override the evaluation budget for the ackley benchmark
python main_NeSTBO.py benchmark=ackley seed=0 benchmark.n_tot=1000
```

## Citation
If you use this code in your research, please cite the following paper:

```
@article{tang2025nest,
  title={NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information},
  author={Tang, Wei-Ting and Kudva, Akshay and Paulson, Joel A},
  journal={arXiv preprint arXiv:2510.05516},
  year={2025}
}
```
