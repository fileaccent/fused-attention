#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fused_attn.cuh"
#include <stdio.h>
#define LAUNCH_WITH_ARGS(head_dim, chunk_size)                      \
    launch_attention_kernel<head_dim, chunk_size>(                          \
        batch_size, seq_len_q, seq_len_k, num_features,                     \
        (half_t*)queries.data_ptr<at::Half>(),                              \
        (half_t*)keys.data_ptr<at::Half>(),                                 \
        (half_t*)values.data_ptr<at::Half>(),                               \
        (half_t*)(mask.has_value() ? mask->data_ptr<at::Half>() : nullptr), \
        (half_t*)output.data_ptr<at::Half>()                                \
    )
#define LAUNCH_WITH_ARGS_TRANS(head_dim, chunk_size)                      \
    launch_attention_trans_kernel<head_dim, chunk_size>(                          \
        batch_size, seq_len_q, seq_len_k, num_features,                     \
        (half_t*)queries.data_ptr<at::Half>(),                              \
        (half_t*)keys.data_ptr<at::Half>(),                                 \
        (half_t*)values.data_ptr<at::Half>(),                               \
        (half_t*)(mask.has_value() ? mask->data_ptr<at::Half>() : nullptr), \
        (half_t*)output.data_ptr<at::Half>()                                \
    )

torch::Tensor attention_forward(
    int head_dim, 
    int chunk_size, 
    torch::Tensor queries, 
    torch::Tensor keys, 
    torch::Tensor values, 
    torch::optional<torch::Tensor> mask
) {
    queries = queries.contiguous();
    keys = keys.contiguous();
    values = values.contiguous();
    // Ensure inputs are of correct shape, dtype, and contiguous
    TORCH_CHECK(queries.dtype() == torch::kFloat16 && 
                queries.device().type() == torch::kCUDA && 
                queries.is_contiguous() &&
                (queries.dim() == 3 || queries.dim() == 4));
                
    TORCH_CHECK(keys.dtype() == torch::kFloat16 && 
                keys.device().type() == torch::kCUDA && 
                keys.is_contiguous() &&
                (keys.dim() == 3 || keys.dim() == 4));

    TORCH_CHECK(values.dtype() == torch::kFloat16 && 
                values.device().type() == torch::kCUDA && 
                values.is_contiguous() &&
                (values.dim() == 3 || values.dim() == 4));

    TORCH_CHECK(!mask.has_value() ||
                (mask->dtype() == torch::kFloat16 && 
                 mask->device().type() == torch::kCUDA && 
                 mask->is_contiguous() &&
                 ((mask->dim() == 2) || (mask->dim() == 4 && mask -> sizes()[0] == 1 && mask -> sizes()[1] == 1))
                ));

    TORCH_CHECK(!(head_dim & (head_dim - 1)) && 16 <= head_dim && head_dim <= 128);
    TORCH_CHECK(!(chunk_size & (chunk_size - 1)) && 16 <= chunk_size && chunk_size <= 128 && chunk_size <= 2 * head_dim);
    // TORCH_CHECK(!(chunk_size == 16 && head_dim == 128));

    // Retrieve input dimensions
    const uint32_t batch_size = queries.size(0);
    const uint32_t seq_len_q = queries.size(1);
    uint32_t num_features = queries.size(2);
    if (queries.dim() == 4) {
        num_features = queries.size(2) * queries.size(3);
    }
    const uint32_t seq_len_k = keys.size(1);
    // Check if other tensors have the same shape/*
    // seq_len_q may not equal to seq_len_k and seq_len_v
    // TORCH_CHECK(keys.size(0) == batch_size && 
    //             keys.size(1) == seq_len_k &&
    //             keys.size(2) == num_features);

    // TORCH_CHECK(values.size(0) == batch_size && 
    //             values.size(1) == seq_len_k &&
    //             values.size(2) == num_features);

    // TORCH_CHECK(seq_len % chunk_size == 0 &&
    //             num_features % head_dim == 0);
    
    // Allocate output tensor
    auto output = torch::empty(
        {batch_size, seq_len_q, num_features},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
    );
    
    // auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (head_dim)
    {
    case 16:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS(16, 16); break;
        case 32: LAUNCH_WITH_ARGS(16, 32); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 32:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS(32, 16); break;
        case 32: LAUNCH_WITH_ARGS(32, 32); break;
        case 64: LAUNCH_WITH_ARGS(32, 64); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 64:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS(64, 16); break;
        case 32: LAUNCH_WITH_ARGS(64, 32); break;
        case 64: LAUNCH_WITH_ARGS(64, 64); break;
        case 128: LAUNCH_WITH_ARGS(64, 128); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 128:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS(128, 16); break;
        case 32: LAUNCH_WITH_ARGS(128, 32); break;
        case 64: LAUNCH_WITH_ARGS(128, 64); break;
        case 128: LAUNCH_WITH_ARGS(128, 128); break;
        default: TORCH_CHECK(false);
        }
        break;
    default: TORCH_CHECK(false);
    }

    // Check if kernel invocation done well
    CHECK_LAST_CUDA_ERROR();

    return output;
}

torch::Tensor attention_forward_trans(
    int head_dim, 
    int chunk_size, 
    torch::Tensor queries, 
    torch::Tensor keys, 
    torch::Tensor values, 
    torch::optional<torch::Tensor> mask
) {
    queries = queries.contiguous();
    keys = keys.contiguous();
    values = values.contiguous();
    // Ensure inputs are of correct shape, dtype, and contiguous
    TORCH_CHECK(queries.dtype() == torch::kFloat16 && 
                queries.device().type() == torch::kCUDA && 
                queries.is_contiguous() &&
                (queries.dim() == 3 || queries.dim() == 4));
                
    TORCH_CHECK(keys.dtype() == torch::kFloat16 && 
                keys.device().type() == torch::kCUDA && 
                keys.is_contiguous() &&
                (keys.dim() == 3 || keys.dim() == 4));

    TORCH_CHECK(values.dtype() == torch::kFloat16 && 
                values.device().type() == torch::kCUDA && 
                values.is_contiguous() &&
                (values.dim() == 3 || values.dim() == 4));

    TORCH_CHECK(!mask.has_value() ||
                (mask->dtype() == torch::kFloat16 && 
                 mask->device().type() == torch::kCUDA && 
                 mask->is_contiguous() &&
                 ((mask->dim() == 2) || (mask->dim() == 4 && mask -> sizes()[0] == 1 && mask -> sizes()[1] == 1))
                ));

    // Retrieve input dimensions
    const uint32_t batch_size = queries.size(0);
    const uint32_t num_heads = queries.size(1);
    const uint32_t seq_len_q = queries.size(2);
    // const uint32_t head_dim = queries.size(3);
    const uint32_t num_features = num_heads * head_dim;
    // if (queries.dim() == 4) {
    //     num_features = queries.size(2) * queries.size(3);
    // }
    const uint32_t seq_len_k = keys.size(2);
    
    // Allocate output tensor
    auto output = torch::empty(
        {batch_size, seq_len_q, num_heads, head_dim},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA)
    );
    
    // auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (head_dim)
    {
    case 16:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS_TRANS(16, 16); break;
        case 32: LAUNCH_WITH_ARGS_TRANS(16, 32); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 32:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS_TRANS(32, 16); break;
        case 32: LAUNCH_WITH_ARGS_TRANS(32, 32); break;
        case 64: LAUNCH_WITH_ARGS_TRANS(32, 64); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 64:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS_TRANS(64, 16); break;
        case 32: LAUNCH_WITH_ARGS_TRANS(64, 32); break;
        case 64: LAUNCH_WITH_ARGS_TRANS(64, 64); break;
        case 128: LAUNCH_WITH_ARGS_TRANS(64, 128); break;
        default: TORCH_CHECK(false);
        }
        break;
    case 128:
        switch (chunk_size)
        {
        case 16: LAUNCH_WITH_ARGS_TRANS(128, 16); break;
        case 32: LAUNCH_WITH_ARGS_TRANS(128, 32); break;
        case 64: LAUNCH_WITH_ARGS_TRANS(128, 64); break;
        case 128: LAUNCH_WITH_ARGS_TRANS(128, 128); break;
        default: TORCH_CHECK(false);
        }
        break;
    default: TORCH_CHECK(false);
    }

    // Check if kernel invocation done well
    CHECK_LAST_CUDA_ERROR();
    // output = output.view({batch_size, seq_len_q, num_heads, head_dim});
    // output = output.transpose(1, 2);
    // output = output.contiguous().view({batch_size, seq_len_q, num_features});
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward",
          &attention_forward,
          py::arg("head_dim"),
          py::arg("chunk_size"),
          py::arg("queries"),
          py::arg("keys"),
          py::arg("values"),
          py::arg("mask") = torch::optional<torch::Tensor>(),
          "Fused multihead attention");
    m.def("attention_forward_trans",
          &attention_forward_trans,
          py::arg("head_dim"),
          py::arg("chunk_size"),
          py::arg("queries"),
          py::arg("keys"),
          py::arg("values"),
          py::arg("mask") = torch::optional<torch::Tensor>(),
          "Fused multihead attention");
}