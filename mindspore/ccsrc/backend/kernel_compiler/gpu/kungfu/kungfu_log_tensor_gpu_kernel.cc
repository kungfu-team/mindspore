#include <algorithm>
#include <numeric>

#include "backend/kernel_compiler/gpu/kungfu/kungfu_log_tensor_gpu_kernel.h"
#include "utils/system/crc32c.h"

namespace mindspore::kernel {

template <typename T>
struct dtype_name {
  const char *operator()() const { return "?"; }
};

template <>
struct dtype_name<float> {
  const char *operator()() const { return "f32"; }
};

template <>
struct dtype_name<int32_t> {
  const char *operator()() const { return "i32"; }
};

template <>
struct dtype_name<int64_t> {
  const char *operator()() const { return "i64"; }
};

template <typename T>
void dbg_log_tensor<T>::operator()(const T *input_addr, T *output_addr, size_t count, cudaStream_t stream) {
  const size_t size = count * sizeof(T);
  cudaMemcpyAsync(output_addr, input_addr, size, cudaMemcpyDeviceToDevice, stream);
  std::vector<T> x(count);

  cudaMemcpyAsync(x.data(), output_addr, size, cudaMemcpyDeviceToHost, stream);
  auto result = cudaStreamSynchronize(stream);
  if (result != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamSynchronize failed";
  } else {
    uint32_t crc = system::Crc32c::GetMaskCrc32cValue((char *)x.data(), size);
    fprintf(stderr, "dbg_log_tensor: crc32:    0x%08x, %s[%d], \n", crc, dtype_name<T>()(), (int)count);
  }
}

template struct dbg_log_tensor<float>;
template struct dbg_log_tensor<int32_t>;
template struct dbg_log_tensor<int64_t>;

// MS_REG_GPU_KERNEL_ONE(KungFuLogTensor,
// KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
//                       KungFuLogTensorGpuKernel, kungfu::float16)
MS_REG_GPU_KERNEL_ONE(KungFuLogTensor, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      KungFuLogTensorGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(KungFuLogTensor, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      KungFuLogTensorGpuKernel, int32_t)
MS_REG_GPU_KERNEL_ONE(KungFuLogTensor, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      KungFuLogTensorGpuKernel, int64_t)
}  // namespace mindspore::kernel
