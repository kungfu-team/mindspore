#include <thread>

#include "pybind_api/api_register.h"
#include "backend/kernel_compiler/cpu/kungfu_common.h"

std::unique_ptr<kungfu::Peer> _kungfu_peer;

bool show_kungfu_debug_log() { return getenv("KUNGFU_MINDSPORE_DEBUG") != nullptr; }

bool _show_kungfu_debug_log = show_kungfu_debug_log();

namespace mindspore {
namespace kernel {
void kungfu_init() {
  log_func_call(__func__);
  _kungfu_peer.reset(new kungfu::Peer);
}

void kungfu_finalize() {
  log_func_call(__func__);
  _kungfu_peer.reset(nullptr);
}

void LOG_InitKernel(const std::string &kernel_name) {
  if (_show_kungfu_debug_log) {
    std::cerr << kernel_name << "::InitKernel called"        //
              << "in thread " << std::this_thread::get_id()  //
              << std::endl;
  }
}

void LOG_Kernel_Launch(const std::string &kernel_name, const std::vector<AddressPtr> &inputs,
                       const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) {
  if (_show_kungfu_debug_log) {
    std::cerr << kernel_name << "::Launch called with "      //
              << inputs.size() << " inputs and "             //
              << outputs.size() << " outputs and "           //
              << workspace.size() << " workspaces"           //
              << "in thread " << std::this_thread::get_id()  //
              << std::endl;
  }
}

int kungfu_current_rank() { return _kungfu_peer->Rank(); }

int kungfu_current_cluster_size() { return _kungfu_peer->Size(); }

REGISTER_PYBIND_DEFINE(KungFu_, ([](py::module *m) {
                         m->def("kungfu_init", &kungfu_init);
                         m->def("kungfu_finalize", &kungfu_finalize);
                         m->def("kungfu_current_rank", &kungfu_current_rank);
                         m->def("kungfu_current_cluster_size", &kungfu_current_cluster_size);
                       }));
}  // namespace kernel
}  // namespace mindspore
