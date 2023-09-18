# Fused attention (Note: This repository currently only supports hip language and does not support cuda.)
The repository is optimized based on ![original repository](https://github.com/kst179/fused-attention.git).
## Description

Specially optimize the attention operation during llama2 inference process. This repository is designed to solve the following problems:
1. In llama2 inference, the small value of seq_len_q causes most of the attention optimization to fail. For example: the prerequisite for flash-attention is that seq_len_q is much larger than head_dim. Most libraries have not optimized this phenomenon.
2. In llama2 inference, the values of seq_len_q and seq_len_k are not the same. Most libraries don't do adaptation.

To install it as torch extention use: (Currently only supports AMD gfx90a architecture.)

```bash
$ python setup.py install
```

Then you can use the extention in following way:
```python
import torch        # should import torch first
from fused_attn import attention_forward

head_dim = 128      # head dim should be a power of 2 and between 16 and 128
chunk_size = 128    # chunk size should also be a power of 2 greater than 16 but less than 2*head_dim,
                    # also if head_dim=128, chunk_size=16 is prohibited (due to the implementation)
                    # I believe the best choice is to use chunk_size == head_dim

q, k, v = ...       # Tensors should be of shape (batch_size, sequence_len, feature_size)
                    # sequence_len should be divisible by chunk_size (if not then should be padded with zeroes), feature_size - by head_dim

m = ...             # optional mask of size (batch_size, sequence_len, sequence_len) filled with zeroes if 
                    # query-key pair should not be muted and -inf else 

output = attention_forward(head_dim, chunk_size, q, k, v, m)
```

There also a CMake project which just builds a simple test that everything is working. You can build and run it with
```bash
$ mkdir build 
$ cd build 
$ cmake ..
$ make
$ ./test
```

- note: If you encounter other problems, you can check this URL https://github.com/ROCmSoftwarePlatform/rocWMMA/issues/239.