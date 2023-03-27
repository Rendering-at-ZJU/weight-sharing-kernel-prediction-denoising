#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


namespace {

__device__ inline float3 operator*(const float &a, const float3 &b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ inline float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3& operator+=(float3& a, float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template <typename scalar_t>
__global__ void CUSTOM_WSKP_auto_cuda_forward_kernel(
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> albedo,
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> guidemap,
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> alpha,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output) {
    const int B = input.size(0), H = input.size(2), W = input.size(3);
    const int C = 6;
    // batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_h = c / W;
    const int idx_w = c % W;
    if (n >= B || idx_h >= H || idx_w >= W) return;

    // //batch index
    // const int n = blockIdx.z;
    // // column index
    // const int idx_h = blockIdx.x * blockDim.x + threadIdx.x;
    // const int idx_w = blockIdx.y * blockDim.y + threadIdx.y;
    

    float weights_sum[C] = {0};
    float3 colors[C] = {0};
    float3 sub_color, final_color = {0};
    float3 sub_albedo = make_float3(albedo[n][0][idx_h][idx_w], albedo[n][1][idx_h][idx_w], albedo[n][2][idx_h][idx_w]);
    
    int h, w;
    float weight;
    for (int i = -C; i <= C; i++) {
        for (int j = -C; j <= C; j++) {
            // min-max is faster than if-continue
            h = max(0, min(idx_h + i, H-1));
            w = max(0, min(idx_w + j, W-1));
                    
            sub_color = make_float3(input[n][0][h][w], input[n][1][h][w], input[n][2][h][w]);

            weight = guidemap[n][C-1][h][w];
            weights_sum[C-1] += weight;
            colors[C-1] += weight * sub_color;
            
            for (int k = C-1; k > 0; k--) {
                if (i >= -k && i <= k && j >= -k && j <= k) {
                    weight = guidemap[n][k-1][h][w];
                    weights_sum[k-1] += weight;
                    colors[k-1] += weight * sub_color;
                }
            }
        }
    }

    for (int k = 0; k < C; k++) {
        final_color += (alpha[n][k][idx_h][idx_w] / weights_sum[k]) * colors[k] ;
    }
    final_color = sub_albedo * final_color;
        
    output[n][0][idx_h][idx_w] = final_color.x;
    output[n][1][idx_h][idx_w] = final_color.y;
    output[n][2][idx_h][idx_w] = final_color.z;
}
    
} // namespace
    

torch::Tensor CUSTOM_WSKP_auto_cuda_forward(
        torch::Tensor input,
        torch::Tensor albedo,
        torch::Tensor guidemap,
        torch::Tensor alpha) {
    
    const auto batch_size = input.size(0);
    const auto height = input.size(2);
    const auto width = input.size(3);

    auto output = torch::zeros_like(input);

    const int threads = 512;
    // const dim3 threadss(threads, threads, 1);
    const dim3 blocks((height * width + threads - 1) / threads, batch_size);

    // const int threads = 128;
    // const dim3 threadss(threads, threads, 1);
    // const dim3 blocks((height + threads - 1) / threads, (width + threads - 1) / threads, batch_size);


    AT_DISPATCH_FLOATING_TYPES(input.type(), "CUSTOM_WSKP_auto_forward_cuda", ([&] {
        CUSTOM_WSKP_auto_cuda_forward_kernel<scalar_t><<<blocks, threads>>> (
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            albedo.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            guidemap.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            alpha.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
    }));

    return output;
}