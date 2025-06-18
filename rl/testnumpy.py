import numpy as np
test = np.array([[1,23,21,24],[32,46,86,23]])
def func(x, bits):
    # bit_width = 8  # You can change to 16, 32, etc.
    bit_array = ((x[..., None] & (1 << np.arange(bits))) > 0).astype(np.uint8)
    return bit_array
print(func(test,8))