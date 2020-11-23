#pragma once
#include <kungfu/nccl/helper.hpp>

class KungFuWrapper {
   public:
    KungFuWrapper();
    ~KungFuWrapper();
};

extern std::unique_ptr<kungfu::NCCLHelper> _kungfu_nccl_helper;
extern std::unique_ptr<KungFuWrapper> _kungfu_wrapper;

namespace mindspore {
namespace kernel {
extern void init_kungfu_nccl_once();
extern void finalize_kungfu_nccl();

template <typename T>
kungfu::Workspace make_kungfu_workspace(const T *input, T *output, int count) {
    return {
      .sendbuf = input,
      .recvbuf = output,
      .count = count,
      .dtype = kungfu::type_encoder::value<T>(),
    };
}
}  // namespace kernel
}  // namespace mindspore
