import os
import sys

# The package is ref/bf16; the tests import it as `bf16`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
