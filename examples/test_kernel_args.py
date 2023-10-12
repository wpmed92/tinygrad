from tinygrad.tensor import Tensor

if __name__ == "__main__":
    ret = Tensor.rand(2,2)
    for _ in range(8): ret = ret + Tensor.rand(2,2)
    ret.realize()
