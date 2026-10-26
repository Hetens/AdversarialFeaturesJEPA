# AdversarialFeaturesJEPA

This repo explores adversarial slow features and the inverse dynamics loss effectiveness in JEPA models.

## Data

OGBench Cube, single-cube play split (`cube-single-play-v0`). The local UV environment installs `ogbench`; running the loader downloads train/val into `Data/ogbench` and writes a small inspectable slice plus a few rendered frames under `Data/preview`.

```bash
uv sync
uv run python DataLoading/load_og.py
```

```python
from DataLoading.load_og import load_og_bench

env, train_dataset, val_dataset = load_og_bench()
```

```bibtex
@inproceedings{ogbench_park2025,
  title={OGBench: Benchmarking Offline Goal-Conditioned RL},
  author={Park, Seohong and Frans, Kevin and Eysenbach, Benjamin and Levine, Sergey},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
}
```

