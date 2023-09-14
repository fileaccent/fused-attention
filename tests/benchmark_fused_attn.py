import torch
import time

import fused_attn

from naive_implementation import attention
import numpy as np
def bench():
    batch_size = 1
    num_heads = 1
    head_dim = 16
    chunk_size = 16
    num_features = num_heads * head_dim
    sequence_len_q = 1
    sequence_len_k = 17
    num_batchs = 1
    
    torch.random.manual_seed(179)

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # queries = torch.randint(-100, 100, (batch_size, sequence_len_q, num_features), device="cuda",dtype=torch.half)
    # keys =    torch.randint(-100, 100, (batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
    # values =  torch.randint(-100, 100, (batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
    queries = torch.rand((batch_size, sequence_len_q, num_features), device="cuda",dtype=torch.half)
    keys =    torch.rand((batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
    values =  torch.randn((batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)

    # # time.sleep(10) # cooldown gpu after tensors' initialization
    # _ = attention(queries, keys, values, head_dim)
    # start = time.time()
    # for i in range(num_batchs):
    #     _ = attention(queries, keys, values, head_dim)
    # torch.cuda.synchronize()
    # end = time.time()

    # naive_ms = (end - start) / num_batchs * 1000

    # # time.sleep(20) # cooldown gpu after calculations
    # _ = fused_attn.attention_forward(head_dim, chunk_size, queries, keys, values)
    # start = time.time()
    # for _ in range(num_batchs):
    #     _ = fused_attn.attention_forward(head_dim, chunk_size, queries, keys, values)
    # torch.cuda.synchronize()
    # end = time.time()

    # fused_ms = (end - start) / num_batchs * 1000

    # print(f"Naive {naive_ms:.4f} ms")
    # print(f"Fused {fused_ms:.4f} ms")
    
    output = attention(queries, keys, values, head_dim)

    output_fu = fused_attn.attention_forward(head_dim, chunk_size, queries, keys, values)
    
    print("output:", np.array((output)[0][0].cpu()).tolist()[0:16])
    print("output_fu:", np.array((output_fu)[0][0].cpu()).tolist()[0:16])
    print("output_error:", np.array((output - output_fu)[0][0].cpu()).tolist()[0:16])
    absolute_index = (output - output_fu).abs().argmax()
    absolute_error = (output - output_fu).abs().max().item()
    # print(absolute_error)
    # print(queries)
    # print(keys)
    # print(values)
    # print(output)
    relative_index = ((output - output_fu) / torch.max(torch.abs(output), torch.abs(output_fu))).abs().argmax()

    relative_error = ((output - output_fu) / torch.max(torch.abs(output), torch.abs(output_fu))).abs().max().item()
    # print(relative_error)
    # # print(output)
    # # print(output_fu)
    print(absolute_error, relative_error, output.flatten()[absolute_index], output_fu.flatten()[absolute_index], output.flatten()[relative_index], output_fu.flatten()[relative_index])
if __name__ == "__main__":
    bench()
