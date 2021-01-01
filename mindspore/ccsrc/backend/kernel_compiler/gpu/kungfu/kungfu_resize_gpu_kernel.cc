#include <cuda_runtime.h>

#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_resize_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class cuda_var {
  T *ptr_;

 public:
  cuda_var(T *ptr) : ptr_(ptr) {}

  T get() const {
    T x;
    auto ret = cudaMemcpy(&x, ptr_, sizeof(T), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
      MS_LOG(ERROR) << "cudaMemcpy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    }
    return x;
  }

  void set(const T &x) {
    auto ret = cudaMemcpy(ptr_, &x, sizeof(T), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
      MS_LOG(ERROR) << "cudaMemcpy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    }
  }
};

template <typename T>
cuda_var<T> make_cuda_var(void *ptr) {
  return cuda_var<T>(reinterpret_cast<T *>(ptr));
}

bool KungFuResizeGpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  LOG_Kernel_Launch("KungFuResizeGpuKernel", inputs, workspace, outputs);
  auto cuda_new_size = make_cuda_var<uint32_t>(inputs.at(0)->addr);
  auto cuda_changed = make_cuda_var<bool>(outputs.at(0)->addr);
  auto cuda_detached = make_cuda_var<bool>(outputs.at(1)->addr);
  bool changed, detached;
  _kungfu_peer->ResizeCluster(cuda_new_size.get(), &changed, &detached);
  cuda_changed.set(changed);
  cuda_detached.set(detached);
  std::cerr << "resized" << std::endl;
  if (changed && !detached) {
    nccl_controller_->ReInit(_kungfu_peer.get());
  }
  std::cerr << "nccl_controller_->ReInit done" << std::endl;
  return true;
}

MS_REG_GPU_KERNEL_REGULAR(
  KungFuResize,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  KungFuResizeGpuKernel)
}  // namespace kernel
}  // namespace mindspore
