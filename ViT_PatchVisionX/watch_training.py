"""Live monitor for a training run.   python watch_training.py [experiment] [seed]   (defaults: full 42)
Stop it with Ctrl+C: this never affects the training itself."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # ViT_PatchVisionX/
sys.path.insert(0, str(ROOT))

from patchvision.monitor import watch

if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "full"
    seed = sys.argv[2] if len(sys.argv) > 2 else "42"
    watch(ROOT / "results_protocol" / exp / f"seed{seed}", interval=2.0, rows=15)
