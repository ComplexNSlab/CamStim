import os
import numpy as np
__version__ = '3.0.1'

# NumPy 1.24+ removed legacy aliases used in this vendored package.
if 'bool' not in np.__dict__:
    np.bool = np.bool_
if 'int' not in np.__dict__:
    np.int = int
if 'float' not in np.__dict__:
    np.float = float

def test():
    import pytest
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(curr_dir, 'test')
    test_dir = test_dir.replace('\\', '/')
    pytest.main(test_dir)