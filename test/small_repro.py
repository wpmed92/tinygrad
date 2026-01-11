from tinygrad import Tensor, dtypes
import numpy as np

#1023 is OK
for n in [1023]:
  t = Tensor.ones(1024, 1024, n, dtype=dtypes.float32)
  result = t.numpy()
  correct = np.sum(result == 1.0)
  print(f"{t.nbytes()} bytes: {correct}/{t.numel()} correct")