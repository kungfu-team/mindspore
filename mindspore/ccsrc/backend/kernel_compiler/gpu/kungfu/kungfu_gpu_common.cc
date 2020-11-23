#include <mutex>

#include <kungfu/nccl/helper.hpp>

// #include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
// #include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "mindspore/core/utils/log_adapter.h"

std::unique_ptr<kungfu::NCCLHelper> _kungfu_nccl_helper;

namespace mindspore {
namespace kernel {
void init_kungfu_nccl_once() {
  static std::mutex mu;
  std::lock_guard<std::mutex> _(mu);
  if (_kungfu_nccl_helper.get() == nullptr) {
    _kungfu_nccl_helper.reset(new kungfu::NCCLHelper);
  }
}

void finalize_kungfu_nccl() { _kungfu_nccl_helper.reset(nullptr); }

/*
class KungFuFinalizeNcclCPUKernel : public CPUKernel {
   public:
    KungFuFinalizeNcclCPUKernel() { MS_LOG(WARNING) << __func__ << " called"; }

    void InitKernel(const CNodePtr &kernel_node) override {}

    bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                const std::vector<AddressPtr> &outputs) override {
        MS_LOG(WARNING) << "KungFuFinalizeNcclCPUKernel" << __func__ << " called";
        finalize_kungfu_nccl();
    }
};

MS_REG_CPU_KERNEL(KungFuFinalizeNccl, KernelAttr(), KungFuFinalizeNcclCPUKernel);
*/
}  // namespace kernel
}  // namespace mindspore
