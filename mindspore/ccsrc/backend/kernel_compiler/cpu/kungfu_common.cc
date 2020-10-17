#include "backend/kernel_compiler/cpu/kungfu_common.h"

std::unique_ptr<kungfu::Peer> _kungfu_peer;

bool show_kungfu_debug_log() { return getenv("KUNGFU_MINDSPORE_DEBUG") != nullptr; }

bool _show_kungfu_debug_log = show_kungfu_debug_log();

namespace mindspore {
namespace kernel {
// TODO: mutex?
void init_kungfu_once() {
  if (_kungfu_peer.get() == nullptr) {
    _kungfu_peer.reset(new kungfu::Peer);
  }
}

void LOG_InitKernel(const std::string &kernel_name) {
  if (_show_kungfu_debug_log) {
    std::cerr << kernel_name << "::InitKernel called" << std::endl;
  }
}

void LOG_Kernel_Launch(const std::string &kernel_name, const std::vector<AddressPtr> &inputs,
                       const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) {
  if (_show_kungfu_debug_log) {
    std::cerr << kernel_name << "::Launch called with "  //
              << inputs.size() << " inputs and "         //
              << outputs.size() << " outputs and "       //
              << workspace.size() << " workspaces" << std::endl;
  }
}
}  // namespace kernel
}  // namespace mindspore
