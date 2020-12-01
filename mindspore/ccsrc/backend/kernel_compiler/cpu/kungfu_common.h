#pragma once
#include <memory>
#include <iostream>
#include <thread>

#include <kungfu.h>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/kungfu_profiler.h"

extern std::unique_ptr<kungfu::Peer> _kungfu_peer;
extern bool _show_kungfu_debug_log;

inline void log_func_call(const char *name) {
  std::cerr << name << " called "                          //
            << "in thread " << std::this_thread::get_id()  //
            << std::endl;
}

namespace mindspore {
namespace kernel {

void LOG_InitKernel(const std::string &kernel_name);

void LOG_Kernel_Launch(const std::string &kernel_name, const std::vector<AddressPtr> &inputs,
                       const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs);

}  // namespace kernel
}  // namespace mindspore
