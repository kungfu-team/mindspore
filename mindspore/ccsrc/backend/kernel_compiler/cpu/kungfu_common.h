#pragma once
#include <memory>
#include <iostream>

#include <kungfu.h>

#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"

extern std::unique_ptr<kungfu::Peer> _kungfu_peer;

namespace mindspore {
namespace kernel {

void init_kungfu_once();

inline void LOG_CALL(const std::string &kernel_name, const std::vector<AddressPtr> &inputs,
                     const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) {
  std::cerr << kernel_name << "::Launch called with "  //
            << inputs.size() << " inputs and "         //
            << outputs.size() << " outputs and "       //
            << workspace.size() << " workspaces" << std::endl;
}

}  // namespace kernel
}  // namespace mindspore
