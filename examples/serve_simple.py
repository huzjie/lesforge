"""Start the OpenAI-compatible server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesforge.config import Config
from lesforge.serving.server import run_server

run_server(Config())
