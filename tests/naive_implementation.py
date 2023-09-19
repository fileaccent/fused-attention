import torch
import torch.nn.functional as F

def attention(queries, keys, values, head_dim):
    batch_size, sequence_len_q, num_features = queries.shape
    sequence_len_k = keys.shape[1]
    num_heads = num_features // head_dim

    queries = queries.view(batch_size, sequence_len_q, num_heads, head_dim).transpose(1, 2)
    keys = keys.view(batch_size, sequence_len_k, num_heads, head_dim).transpose(1, 2)
    values = values.view(batch_size, sequence_len_k, num_heads, head_dim).transpose(1, 2)

    scores = (queries @ keys.mT) / (head_dim ** 0.5)
    scores = torch.softmax(scores.float(), dim=-1).type_as(queries)
    output = scores @ values
    output = output.transpose(1, 2).contiguous().view(batch_size, sequence_len_q, num_features)    


    # scores = torch.matmul(xq_, keys_.transpose(2, 3)) / math.sqrt(self.head_dim)
    #     # if mask is not None:
    #     #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
    #     scores = F.softmax(scores.float(), dim=-1).type_as(xq_)
    #     output = torch.matmul(scores, values_)  # (bs, n_local_heads, seqlen, head_dim)
    return output

def attention_float(queries, keys, values, head_dim, mask=None):
    batch_size, sequence_len_q, num_features = queries.shape
    sequence_len_k = keys.shape[1]
    num_heads = num_features // head_dim

    queries = queries.view(batch_size, sequence_len_q, num_heads, head_dim).transpose(1, 2).float()
    keys = keys.view(batch_size, sequence_len_k, num_heads, head_dim).transpose(1, 2).float()
    values = values.view(batch_size, sequence_len_k, num_heads, head_dim).transpose(1, 2).float()
    # print("mask", mask.shape)
    scores = (queries @ keys.mT) / (head_dim ** 0.5)
    if mask is not None:
        # mask = mask
        scores = scores + mask
    scores = torch.softmax(scores.float(), dim=-1)
    output = scores @ values
    output = output.transpose(1, 2).contiguous().view(batch_size, sequence_len_q, num_features)    


    # scores = torch.matmul(xq_, keys_.transpose(2, 3)) / math.sqrt(self.head_dim)
    #     # if mask is not None:
    #     #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
    #     scores = F.softmax(scores.float(), dim=-1).type_as(xq_)
    #     output = torch.matmul(scores, values_)  # (bs, n_local_heads, seqlen, head_dim)
    return output.half()