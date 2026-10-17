import os
import sys

# run the tests from a source checkout without installing the package
try:
    import pyrenyi  # noqa: F401  (installed)
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
