#include <algorithm>
#include <numeric>

//#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_log_tensor_gpu_kernel.h"

namespace mindspore::kernel {
template <typename T>
void log_tensor(int idx, const kungfu::Workspace &w, cudaStream_t stream) {
  std::vector<T> x(w.count);
  const auto dir = cudaMemcpyDeviceToHost;
  cudaMemcpyAsync(x.data(), w.recvbuf, w.count * sizeof(T), dir, stream);
  auto result = cudaStreamSynchronize(stream);
  if (result != cudaSuccess) {
    MS_LOG(ERROR) << "cudaStreamSynchronize failed";
  }

  const T min = *std::min_element(x.begin(), x.end());
  const T max = *std::max_element(x.begin(), x.end());
  const T sum = std::accumulate(x.begin(), x.end(), static_cast<T>(0));
  const T mean = sum / x.size();
  printf("%s %s[%d] :: [%f, %f] ~ %f, sum: %f\n", __func__, "f32",
         (int)x.size(),  //
         min, max, mean, sum);

  char filename[256];
  sprintf(filename, "%06d-%s-%d.data", idx, "f32", (int)x.size());
  fprintf(stderr, "filename: %s\n", filename);
  FILE *fp = fopen(filename, "wb");
  if (fp == nullptr) {
    MS_LOG(ERROR) << "fopen failed";
  }
  fwrite(x.data(), sizeof(T), x.size(), fp);
  fclose(fp);
}

void log_workspace(int idx, const kungfu::Workspace &w, cudaStream_t stream) {
  if (_show_kungfu_debug_log) {
    std::cerr << __func__ << ", count: " << w.count << std::endl;
  }
  const size_t size = kungfu_type_size(w.dtype) * w.count;
  const auto dir = cudaMemcpyDeviceToDevice;
  // auto result = cudaMemcpy(w.recvbuf, w.sendbuf, size, dir);
  // if (result != cudaSuccess) {
  //     MS_LOG(ERROR) << "cudaMemcpy failed";
  // }
  cudaMemcpyAsync(w.recvbuf, w.sendbuf, size, dir, stream);
  switch (w.dtype) {
    case KungFu_FLOAT:
      log_tensor<float>(idx, w, stream);
      return;
    default:
      // ignore
      return;
  }
}

MS_REG_GPU_KERNEL_ONE(KungFuLogTensor, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      KungFuLogTensorGpuKernel, kungfu::float16)
MS_REG_GPU_KERNEL_ONE(KungFuLogTensor, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      KungFuLogTensorGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(KungFuLogTensor, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      KungFuLogTensorGpuKernel, int32_t)
}  // namespace mindspore::kernel
