#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_broadcast_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(KungFuBroadcast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      KungFuBroadcastGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(KungFuBroadcast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      KungFuBroadcastGpuKernel, int32_t)
}  // namespace kernel
}  // namespace mindspore
