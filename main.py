import os
import sys
import traceback

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """
    Hydra entry point
    """

    # Adjust sys.path to include the 'src' directory
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    _SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

    # Lazy import to avoid expensive imports during hydra job submission
    from optimization_loop_NeSTBO import main

    try:
       
        alg = main(config)
        X, Y, min_obj_list = alg.exec_alg()
        
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    sys.argv.append("benchmark=griewank")
    main()
