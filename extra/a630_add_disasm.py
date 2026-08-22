import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad import Tensor, dtypes

N = 16

a = Tensor.empty(N, dtype=dtypes.float32)
b = Tensor.empty(N, dtype=dtypes.float32)

c = a + b

c.realize()


