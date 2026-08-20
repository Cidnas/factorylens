import sys
from pathlib import Path

print("Working directory:", Path.cwd())
print("Executed file:", Path(__file__).resolve())

print("Python import paths:")
for path in sys.path:
    print("  ", path)