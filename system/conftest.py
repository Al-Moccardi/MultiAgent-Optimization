# Ensures `import shared` / `import part1_allocation` work under pytest from root.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
