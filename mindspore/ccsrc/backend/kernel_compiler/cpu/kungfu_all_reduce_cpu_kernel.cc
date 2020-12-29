#include "backend/kernel_compiler/cpu/kungfu_all_reduce_cpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_CPU_KERNEL_T(KungFuAllReduce, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    KungFuAllReduceCPUKernel, float);
MS_REG_CPU_KERNEL_T(KungFuAllReduce, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    KungFuAllReduceCPUKernel, int32_t);
}  // namespace kernel
}  // namespace mindspore
