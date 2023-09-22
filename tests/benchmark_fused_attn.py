import torch
import time

import fused_attn

from naive_implementation import attention, attention_float
import numpy as np
def bench():
    batch_size = 4
    sequence_len_q = 1
    sequence_len_k = 112
    num_heads = 20
    head_dim = 128
    chunk_size = 16
    num_features = num_heads * head_dim
    num_batchs = 128    
    
    torch.random.manual_seed(179)

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    for i in range(1):
        # queries = torch.randint(-30, 30, (batch_size, sequence_len_q, num_features), device="cuda",dtype=torch.half)
        # keys =    torch.randint(-30, 30, (batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
        # values =  torch.randint(-30, 30, (batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
        # queries = torch.ones((batch_size, sequence_len_q, num_features), device="cuda",dtype=torch.half)
        # keys =    torch.ones((batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
        # values =  torch.ones((batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
        queries = torch.randn((batch_size, sequence_len_q, num_features), device="cuda",dtype=torch.half)
        keys =    torch.randn((batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
        values =  torch.randn((batch_size, sequence_len_k, num_features), device="cuda",dtype=torch.half)
        mask = None
        # queries = torch.load('/data/zhaorong/code/fused-attention/xq_fu.pt').cuda().half()
        # keys = torch.load('/data/zhaorong/code/fused-attention/keys_fu.pt').cuda().half()
        # values = torch.load('/data/zhaorong/code/fused-attention/values_fu.pt').cuda().half()
        # mask = torch.load('/data/zhaorong/code/fused-attention/mask_fu.pt').cuda().half()
        # print(queries.device)
        # print(keys.device)
        # print(values.device)
        # # print(batch_size, sequence_len_q, sequence_len_k, num_heads, head_dim)
        # # time.sleep(10) # cooldown gpu after tensors' initialization

        # # time.sleep(20) # cooldown gpu after calculations
        # _ = fused_attn.attention_forward(head_dim, chunk_size, queries, keys, values)
        # start = time.time()
        # for _ in range(num_batchs):
        #     _ = fused_attn.attention_forward(head_dim, chunk_size, queries, keys, values)
        # torch.cuda.synchronize()
        # end = time.time()

        # fused_ms = (end - start) / num_batchs * 1000

        # _ = fused_attn.attention_forward_trans(head_dim, chunk_size, queries, keys, values)
        # start = time.time()
        # for _ in range(num_batchs):
        #     _ = fused_attn.attention_forward_trans(head_dim, chunk_size, queries, keys, values)
        # torch.cuda.synchronize()
        # end = time.time()

        # fused_trans_ms = (end - start) / num_batchs * 1000
        
        # _ = attention_float(queries, keys, values, head_dim, mask)
        # start = time.time()
        # for i in range(num_batchs):
        #     _ = attention_float(queries, keys, values, head_dim, mask)
        # torch.cuda.synchronize()
        # end = time.time()

        # naive_ms = (end - start) / num_batchs * 1000

        # print(f"Naive {naive_ms:.4f} ms")
        # print(f"Fused {fused_ms:.4f} ms")
        # print(f"Fused_trans {fused_trans_ms:.4f} ms")
        
        output = attention_float(queries, keys, values, head_dim, mask)

        output_fu = fused_attn.attention_forward(head_dim, chunk_size, queries, keys, values, mask)
        # output_fu_trans = fused_attn.attention_forward_trans(head_dim, chunk_size, queries, keys, values, mask)
        
        # print("output:", output)
        # print("output_fu:", output_fu)
        # # print("output_fu_trans:", output_fu_trans)
        # # print("output_error:", np.array((output - output_fu)[0][0].cpu()).tolist()[0:16])
        # absolute_index = (output - output_fu).abs().argmax()
        absolute_error = (output - output_fu).abs().max().item()
        # absolute_trans_error = (output - output_fu_trans).abs().max().item()
        print(absolute_error)
        # # print(queries)
        # # print(keys)
        # # print(values)
        # # print(output)
        # relative_index = ((output - output_fu) / torch.max(torch.abs(output), torch.abs(output_fu))).abs().argmax()

        # relative_error = ((output - output_fu) / torch.max(torch.abs(output), torch.abs(output_fu))).abs().max().item()
        # # print(relative_error)
        # print(output)
        # print(output_fu)
        # # print(output_fu_trans)
        # if (torch.any(torch.isnan(output_fu))):
        #     print('isnan', torch.any(torch.isnan(output_fu)))
        #     # print("queries:", queries)
        #     # print("keys:", keys)
        #     # print("values:", values)
        #     # print("output:", output)
        #     # print("output_fu:", output_fu)
        #     break
        # # print(absolute_error, relative_error, output.flatten()[absolute_index], output_fu.flatten()[absolute_index], output.flatten()[relative_index], output_fu.flatten()[relative_index])
        # print(absolute_error, absolute_trans_error)
if __name__ == "__main__":
    bench()
