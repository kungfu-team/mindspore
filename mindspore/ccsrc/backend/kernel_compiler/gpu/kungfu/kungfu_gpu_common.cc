#include <mutex>

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
  MS_LOG(ERROR) << "BEGIN " << __func__;
  _kungfu_nccl_helper.reset(new kungfu::NCCLHelper);

  const auto nccl_scope = KungFu_NCCL_GLOBAL;
  // auto nccl_scheduler =
  _kungfu_nccl_helper->EnsureScheduler(nccl_scope);
  auto nccl_controller = _kungfu_nccl_helper->EnsureController(nccl_scope);

  kungfu::Peer *peer = _kungfu_peer.get();
  nccl_controller->InitOnce(peer);
  MS_LOG(ERROR) << "END " << __func__;
}

void kungfu_nccl_finalize() {
  MS_LOG(ERROR) << "BEGIN " << __func__;
  _kungfu_nccl_helper.reset(nullptr);
  MS_LOG(ERROR) << "END " << __func__;
}

void init_kungfu_nccl_once() {
  static std::mutex mu;
  std::lock_guard<std::mutex> _(mu);
  if (_kungfu_nccl_helper.get() == nullptr) {
    _kungfu_nccl_helper.reset(new kungfu::NCCLHelper);
  }
}

void finalize_kungfu_nccl() { _kungfu_nccl_helper.reset(nullptr); }

REGISTER_PYBIND_DEFINE(KungFuNccl, ([](py::module *m) {
                         m->def("kungfu_nccl_init", &kungfu_nccl_init);
                         m->def("kungfu_nccl_finalize", &kungfu_nccl_finalize);
                       }));
}  // namespace kernel
}  // namespace mindspore
