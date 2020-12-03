#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_resize_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_REGULAR(
  KungFuResize,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  KungFuResizeGpuKernel)
}  // namespace kernel
}  // namespace mindspore
