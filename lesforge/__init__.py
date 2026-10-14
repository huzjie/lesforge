"""lesforge - Lookahead Expert Scheduling (LES) for on-device MoE VLMs.

A zero-dependency reference implementation of the two core mechanisms behind
OpenBMB's MiniCPM-V 4.7 (a 35B mixture-of-experts vision-language model that
runs on a phone at ~3 GB of memory):

* **Expert offloading + tiered memory**: only the active experts stay in RAM,
  the rest live on SSD/NAND and are loaded on demand under a strict memory
  budget (the full 35B weights are ~65.6 GiB -- never all resident).

* **Lookahead Expert Scheduling (LES)**: while layer L's experts are computing,
  the router output already tells us which experts layer L+1 will need, so we
  prefetch them and overlap compute with I/O. This hides SSD latency and is the
  core trick behind the ~5x speedup on an iPad Pro M4.

Public API:
    from lesforge import VLMEngine, Config, get_backend, LEScheduler
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .moe.router import TopKRouter
from .scheduler.les import LEScheduler
from .model.engine import VLMEngine
from .backends import get_backend, list_backends

__all__ = [
    "Config",
    "load_config",
    "TopKRouter",
    "LEScheduler",
    "VLMEngine",
    "get_backend",
    "list_backends",
    "__version__",
]
