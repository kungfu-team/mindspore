#include <kungfu/nccl/helper.hpp>

#include "pybind_api/api_register.h"
#include "backend/kernel_compiler/cpu/kungfu_common.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/kungfu/kungfu_gpu_common.h"
#include "mindspore/core/utils/log_adapter.h"

std::unique_ptr<kungfu::NCCLHelper> _kungfu_nccl_helper;

namespace mindspore {
namespace kernel {
void kungfu_nccl_init() {
  log_func_call(__func__);
  const auto nccl_scope = KungFu_NCCL_GLOBAL;
  _kungfu_nccl_helper.reset(new kungfu::NCCLHelper);
  _kungfu_nccl_helper->EnsureScheduler(nccl_scope);
  auto nccl_controller = _kungfu_nccl_helper->EnsureController(nccl_scope);
  kungfu::Peer *peer = _kungfu_peer.get();
  nccl_controller->InitOnce(peer);
}

void kungfu_nccl_finalize() {
  log_func_call(__func__);
  _kungfu_nccl_helper.reset(nullptr);
}

REGISTER_PYBIND_DEFINE(KungFuNccl, ([](py::module *m) {
                         m->def("kungfu_nccl_init", &kungfu_nccl_init);
                         m->def("kungfu_nccl_finalize", &kungfu_nccl_finalize);
                       }));
}  // namespace kernel
}  // namespace mindspore
