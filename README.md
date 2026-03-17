# NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information

This repository contains the code to reproduce the NeST-BO and NeST-BO-sub algorithms proposed in the paper _NeST-BO: Fast Local Bayesian Optimization via Newton-Step Targeting of Gradient and Hessian Information_. 

NeST-BO has been published as a conference paper in AISTATS 2026.

## Running Experiments

Experiments can be run using the `main_NeSTBO.py` and `main_NeSTBO_sub.py` script. You must specify a benchmark to run.

**Basic Command**
```
python main_NeSTBO.py benchmark=<benchmark_name>
python main_NeSTBO_sub.py benchmark=<benchmark_name>
```

*   To see a list of available benchmarks, run `python main_NeSTBO.py`.
*   Adding `seed=<number>` is recommended for reproducibility.
  
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
