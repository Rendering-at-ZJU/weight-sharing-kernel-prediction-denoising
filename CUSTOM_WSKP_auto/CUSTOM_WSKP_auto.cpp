#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor CUSTOM_WSKP_auto_cuda_forward(
    torch::Tensor input,
    torch::Tensor albedo,
    torch::Tensor guidemap,
    torch::Tensor alpha);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor CUSTOM_WSKP_auto_forward(
        torch::Tensor input,
        torch::Tensor albedo,
        torch::Tensor guidemap,
        torch::Tensor alpha) {
    CHECK_INPUT(input);
    CHECK_INPUT(albedo);
    CHECK_INPUT(guidemap);
    CHECK_INPUT(alpha);

    return CUSTOM_WSKP_auto_cuda_forward(input, albedo, guidemap, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &CUSTOM_WSKP_auto_forward, "CUSTOM_WSKP_auto forward (CUDA)");
}