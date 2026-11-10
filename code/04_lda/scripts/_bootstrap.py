"""Put the project root on sys.path so `import config` / `import src...` work
regardless of where the script is launched from."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
