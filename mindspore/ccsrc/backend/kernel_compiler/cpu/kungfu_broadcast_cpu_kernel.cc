#include "backend/kernel_compiler/cpu/kungfu_broadcast_cpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_CPU_KERNEL_T(KungFuBroadcast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    KungFuBroadcastCPUKernel, float);
MS_REG_CPU_KERNEL_T(KungFuBroadcast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    KungFuBroadcastCPUKernel, int32_t);
}  // namespace kernel
}  // namespace mindspore
