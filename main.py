import os
import sys
import traceback

import hydra
from omegaconf import DictConfig

# Optimization loop module and the benchmark key it requires, per algorithm
ALGORITHMS = {
    "nestbo": ("optimization_loop_NeSTBO", "M"),
    "nestbo_sub": ("optimization_loop_NeSTBO_sub", "target_dim_init"),
}


def run(config: DictConfig):
    """Runs the algorithm selected by config.algorithm on config.benchmark."""

    if config.algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{config.algorithm}'. Choose one of: {', '.join(ALGORITHMS)}"
        )
    module_name, required_key = ALGORITHMS[config.algorithm]
    if required_key not in config.benchmark:
        raise ValueError(
            f"algorithm={config.algorithm} requires 'benchmark.{required_key}', which is not set "
            f"in the '{config.benchmark.name}' config. Add it on the command line, "
            f"e.g. +benchmark.{required_key}=<value>."
        )

    # Adjust sys.path to include the project root and the 'src' directory
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    _SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
    for path in (_PROJECT_ROOT, _SRC_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Lazy import to avoid expensive imports during hydra job submission
    module = __import__(module_name)

    alg = module.main(config)
    return alg.exec_alg()


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """
    Hydra entry point
    """
    try:
        X, Y, min_obj_list = run(config)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
